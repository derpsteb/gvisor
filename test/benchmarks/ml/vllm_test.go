// Copyright 2020 The gVisor Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package ml

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/docker/docker/api/types/mount"
	"gvisor.dev/gvisor/pkg/test/dockerutil"
	"gvisor.dev/gvisor/test/benchmarks/harness"
	"gvisor.dev/gvisor/test/metricsviz"
)

func TestMain(m *testing.M) {
	harness.Init()
	harness.SetFixedBenchmarks()
	os.Exit(m.Run())
}

// BenchmarkVLLM runs a vLLM workload.
func BenchmarkVLLM(b *testing.B) {
	doVLLMTest(b)
}

func doVLLMTest(b *testing.B) {
	serverMachine, err := harness.GetMachine()
	if err != nil {
		b.Fatalf("failed to get machine: %v", err)
	}
	// defer serverMachine.CleanUp()

	b.Run("opt-125", func(b *testing.B) {
		ctx := context.Background()

		b.ResetTimer()
		b.StopTimer()

		for i := 0; i < b.N; i++ {
			serverCtr := serverMachine.GetNativeContainer(ctx, b)
			defer metricsviz.FromContainerLogs(ctx, b, serverCtr)
			// defer serverCtr.CleanUp(ctx)
			if err := harness.DropCaches(serverMachine); err != nil {
				b.Skipf("failed to drop caches: %v. You probably need root.", err)
			}

			// Run vllm.
			runOpts := dockerutil.GPURunOpts()
			runOpts.CpusetCpus = "0"
			runOpts.Image = "benchmarks/vllm"
			runOpts.Env = []string{"PYTHONPATH=$PYTHONPATH:/vllm"}

			if err := serverCtr.Spawn(ctx, runOpts); err != nil {
				b.Errorf("failed to run container: %v", err)
			}
			if out, err := serverCtr.WaitForOutput(ctx, "Uvicorn running on http://0.0.0.0:8000", 10*time.Minute); err != nil {
				b.Fatalf("failed to start vllm model: %v %s", err, out)
			}

			clientMachine, err := harness.GetMachine()
			if err != nil {
				b.Fatalf("failed to get machine: %v", err)
			}
			// defer clientMachine.CleanUp()
			clientCtr := clientMachine.GetNativeContainer(ctx, b)
			// defer clientCtr.CleanUp(ctx)

			b.StartTimer()

			// store vllm logs here
			logsDir := b.TempDir()

			out, err := clientCtr.Run(ctx, dockerutil.RunOpts{
				Links:      []string{serverCtr.MakeLink("vllmctr")},
				CpusetCpus: "0",
				Image:      "benchmarks/vllm",
				Env:        []string{"PYTHONPATH=$PYTHONPATH:/vllm"},
				Mounts: []mount.Mount{
					// The logs dir is used because vllm only outputs json to a file.
					{
						Source: logsDir,
						Target: "/tmp",
						Type:   "bind",
					},
				},
			}, "/vllm/benchmarks/benchmark_serving.py", "--host", "vllmctr", "--model", "/model", "--tokenizer", "/model", "--endpoint", "/v1/completions", "--backend", "openai", "--dataset", "/ShareGPT_V3_unfiltered_cleaned_split.json", "--save-result", "--result-dir", "/tmp")
			if err != nil {
				b.Errorf("failed to run container: %v logs: %s", err, out)
			}

			b.StopTimer()

			metrics, err := parseVLLMJSON(logsDir)
			if err != nil {
				b.Errorf("failed to parse vllm output: %v", err)
			}
			b.ReportMetric(float64(metrics.Completed), "requests")
			b.ReportMetric(metrics.RequestThroughput, "request_throughput")
			b.ReportMetric(metrics.InputThroughput, "input_tok_throughput")
			b.ReportMetric(metrics.OutputThroughput, "output_tok_throughput")
			b.ReportMetric(metrics.MedianTTFTMS, "median_ttft_ms")
			b.ReportMetric(metrics.MediaTPOTMS, "median_tpot_ms")
		}
	})
}

// Modeled after the metrics reported here: https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py#L338-L358
type metrics struct {
	Duration          float64 `json:"duration"`
	Completed         int     `json:"completed"`
	TotalInputTokens  int     `json:"total_input_tokens"`
	TotalOutputTokens int     `json:"total_output_tokens"`
	RequestThroughput float64 `json:"request_throughput"`
	InputThroughput   float64 `json:"input_throughput"`
	OutputThroughput  float64 `json:"output_throughput"`
	MeanTTFTMS        float64 `json:"mean_ttft_ms"`
	MedianTTFTMS      float64 `json:"median_ttft_ms"`
	P99TTFTMS         float64 `json:"p99_ttft_ms"`
	MeanTPOTMS        float64 `json:"mean_tpot_ms"`
	MediaTPOTMS       float64 `json:"median_tpot_ms"`
	P99TPOTMS         float64 `json:"p99_tpot_ms"`
	// InputLens         []int   `json:"input_lens"`
	// OutputLens        []int   `json:"output_lens"`
	// TTFTs             []float64   `json: "ttfts"`
	// ITLS              [][]float64 `json: "itls"`
	// GeneratedTexts    float64 `json: "generated_texts"`
	// Errors            float64 `json: "errors"`
}

// parseVLLMJSON expects a path that contains only one json file.
func parseVLLMJSON(path string) (metrics, error) {
	files, err := os.ReadDir(path)
	if err != nil {
		return metrics{}, fmt.Errorf("failed to read directory: %w", err)
	}

	var jsonPath string
	for _, name := range files {
		if strings.HasSuffix(name.Name(), ".json") {
			jsonPath = filepath.Join(path, name.Name())
			break
		}
	}
	if jsonPath == "" {
		return metrics{}, errors.New("no json file found")
	}

	data, err := os.ReadFile(jsonPath)
	if err != nil {
		return metrics{}, fmt.Errorf("failed to read file: %w", err)
	}

	var vllm metrics
	if err := json.Unmarshal(data, &vllm); err != nil {
		return metrics{}, fmt.Errorf("failed to unmarshal data: %w", err)
	}

	return vllm, nil
}
