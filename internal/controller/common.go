// Copyright (c) IBM Corporation
// SPDX-License-Identifier: MIT

package controller

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"os"
	"os/exec"
	"path"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/go-logr/logr"
	kubeflowv1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	v1 "k8s.io/api/core/v1"
	"mvdan.cc/sh/v3/syntax"
	ctrl "sigs.k8s.io/controller-runtime"
)

type ResourceRequirements struct {
	// These are the total number of workers across all PyTorchJobReplicaSpecs
	Workers int
	// These are the GPUs per replica of PyTorchJobReplicaSpecs
	GPUs  int
	Error string

	// These are fields we auto-generate based on the above
	// These are the replicas of a specific PyTorchJobReplicaSpecs
	Replicas int
	// CanRecommend indicates whether the recommendation engine was able to generate recommendations
	CanRecommend bool
}

type MinGPURecommenderInput struct {
	ModelName       string
	Method          string
	GPUModel        string
	TokensPerSample int
	BatchSize       int
	ModelVersion    string
}

type PropertyDescriptor struct {
	Identifier string `json:"identifier"`
}

type TargetProperty struct {
	Identifier string `json:"identifier,omitempty"`
}

type ObservedProperty struct {
	TargetProperty TargetProperty `json:"targetProperty"`
}

type Measurement struct {
	Value    int              `json:"value,omitempty"`
	Property ObservedProperty `json:"property"`
}

type MeasurementResult struct {
	Measurements []Measurement `json:"measurements,omitempty"`
}

type ConstitutivePropertyValue struct {
	ValueType string         `json:"valueType"`
	Value     any            `json:"value,omitempty"`
	Property  TargetProperty `json:"property"`
}

type Entity struct {
	Identifier                 string                      `json:"identifier"`
	MeasurementResults         []MeasurementResult         `json:"measurement_results,omitempty"`
	ConstitutivePropertyValues []ConstitutivePropertyValue `json:"constitutive_property_values,omitempty"`
}

type MeasurementRequest struct {
	Entities []Entity `json:"entities,omitempty"`
}

func (e *Entity) GetObservedProperties() map[string]int {
	op := make(map[string]int)

	for _, mr := range e.MeasurementResults {
		for _, m := range mr.Measurements {
			op[m.Property.TargetProperty.Identifier] = m.Value
		}
	}

	return op

}

// Converts a word into a string while retaining the quotes
func WordToString(w *syntax.Word) (string, error) {
	if lit := w.Lit(); lit != "" {
		return lit, nil
	}

	if len(w.Parts) == 1 {
		switch p := w.Parts[0].(type) {
		case *syntax.SglQuoted:
			return p.Value, nil
		case *syntax.DblQuoted:
			var sb strings.Builder
			for _, qp := range p.Parts {
				l, ok := qp.(*syntax.Lit)
				if !ok {
					return "", fmt.Errorf("non-literal WordPard %+v", qp)
				}
				sb.WriteString(l.Value)
			}
			return sb.String(), nil
		}
	}
	return "", fmt.Errorf("expected a word with a single part but got %+v", w.Parts)
}

func isAccelerateLaunch(call *syntax.CallExpr) bool {
	if len(call.Args) < 2 {
		return false
	}

	accelerate, err := WordToString(call.Args[0])
	if err != nil || accelerate != "accelerate" {
		return false
	}

	launch, err := WordToString(call.Args[1])
	if err != nil || launch != "launch" {
		return false
	}

	return true
}

// Convert strings into Words
func StringsToWords(
	parser *syntax.Parser,
	texts []string,
) ([]*syntax.Word, error) {

	newWords := make([]*syntax.Word, 0, len(texts))
	for _, tok := range texts {
		// 2) parse token as a single shell word (quotes included)
		var parsed []*syntax.Word
		err := parser.Words(strings.NewReader(tok), func(w *syntax.Word) bool {
			parsed = append(parsed, w)
			return true
		})

		if err != nil {
			return nil, err
		}
		newWords = append(newWords, parsed...)
	}
	return newWords, nil
}

// Convert words into strings preserving quotes where possible
func WordsToQuotedArgs(words []*syntax.Word) ([]string, error) {
	pr := syntax.NewPrinter(syntax.Minify(true))

	out := make([]string, 0, len(words))
	for _, w := range words {
		var b bytes.Buffer
		err := pr.Print(&b, w)

		if err != nil {
			return out, fmt.Errorf("could not convert word %+v to string due to %w", w, err)
		}

		out = append(out, b.String())
	}
	return out, nil
}

// Extracts the values of flags from a commandline
// e.g. ["--foo=hello", "--bar", "world", "--nope", "discard"] for flags ["--foo", "--bar"]
// will return {"foo": "hello", "bar": "world"} f
func ExtractCommandLineOptions(argv []string, flags []string) map[string]string {
	out := map[string]string{}

	skipLastParsed := false

	for i, a := range argv {
		if skipLastParsed {
			skipLastParsed = false
			continue
		}

		if slices.Contains(flags, a) && i+1 < len(argv) {
			out[a] = argv[i+1]
			skipLastParsed = true
			goto next
		}

		for _, name := range flags {
			if strings.HasPrefix(a, name+"=") {
				parts := strings.Split(a, "=")
				out[name] = strings.Join(parts[1:], "=")

				goto next
			}
		}

	next:
	}

	return out
}

// Ensures that commandline contains options with certain values.
// Works for both `--$name=$oldValue` and `--$name $oldValue`.
func UpsertCommandlineArgs(argv []string, replacements map[string]string) []string {
	out := make([]string, 0, len(argv))

	skipLastParsed := false

	for _, a := range argv {
		if skipLastParsed {
			skipLastParsed = false
			continue
		}

		if r, ok := replacements[a]; ok {
			out = append(out, a, r)
			skipLastParsed = true
			delete(replacements, a)

			goto next
		}

		for name, value := range replacements {
			if strings.HasPrefix(a, name+"=") {
				delete(replacements, name)

				out = append(out, fmt.Sprintf("%s=\"%s\"", name, value))
				goto next
			}
		}

		out = append(out, a)
	next:
	}

	for name, value := range replacements {
		out = append(out, fmt.Sprintf("%s=\"%s\"", name, value))
	}

	return out
}

// Patches the arguments of `accelerate launch` and `python -m tuning.sft_trainer` to use
// reference the environment variables that the operator injects.
func PatchAccelerateLaunchCommandline(argv []string) ([]string, error) {
	idxSplit := slices.Index(argv, "-m")

	if idxSplit == -1 {
		idxSplit = slices.Index(argv, "--module")
	}

	if idxSplit == -1 {
		return nil, fmt.Errorf("unable to find the -m or --module delimiter in the accelerate launch commandline")
	}

	// VV: handle the `accelerate launch` args first
	out := UpsertCommandlineArgs(argv[:idxSplit], map[string]string{
		"--num_processes": "$AUTOCONF_NUM_PROCESSES",
		"--num_machines":  "$AUTOCONF_NUM_MACHINES",
	})

	// VV: Then the ones for tuning.sft_trainer
	out = append(out, UpsertCommandlineArgs(argv[idxSplit:], map[string]string{
		"--per_device_train_batch_size": "$AUTOCONF_PER_DEVICE_BATCH_SIZE",
	})...)

	return out, nil
}

func CheckRequestToCalcMinimumResourceRequirements(in MinGPURecommenderInput, url, requestID string, log logr.Logger) (*ResourceRequirements, error) {
	requestURL := url + "/api/latest/actuators/custom_experiments/experiments/min_gpu_recommender/requests"
	requestURL = requestURL + "/" + requestID
	// VV: Now keep polling requestURL using $requestID till we get back a non 404 result

	resp, err := http.Get(requestURL)

	if err != nil {
		return nil, fmt.Errorf("cannot send GET request due to %w", err)
	}

	if resp.StatusCode == 404 {
		return nil, nil
	}

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("got unexpected status %#v", resp.Status)
	}

	body, err := io.ReadAll(resp.Body)

	if err != nil {
		return nil, fmt.Errorf("cannot read the GET response body due to %w", err)
	}

	mr := MeasurementRequest{}

	err = json.Unmarshal(body, &mr)

	if err != nil {
		return nil, fmt.Errorf("cannot unmarshal GET response body: %s due to %w", string(body), err)
	}

	if len(mr.Entities) != 1 {
		return nil, fmt.Errorf("expected exactly one entity in the response, got %d", len(mr.Entities))
	}

	op := mr.Entities[0].GetObservedProperties()

	if op["can_recommend"] != 1 {
		// Return a ResourceRequirements indicating no recommendation is available
		log.Info("Recommendation engine cannot recommend resources", "observedProperties", op)
		return &ResourceRequirements{
			Workers:      0,
			GPUs:         0,
			CanRecommend: false,
		}, nil
	}

	log.Info("Obtained measured properties", "observedProperties", op, "measurementRequests", mr)

	recs := ResourceRequirements{
		Workers:      op["workers"],
		GPUs:         op["gpus"],
		CanRecommend: true,
	}

	return &recs, nil
}

func SendRequestToCalcMinimumResourceRequirements(in MinGPURecommenderInput, url string, log logr.Logger) (string, error) {
	e := Entity{}

	const (
		stringValueType  = "STRING_VALUE_TYPE"
		numericValueType = "NUMERIC_VALUE_TYPE"
	)

	stringValues := map[string]string{
		"model_name":    in.ModelName,
		"method":        in.Method,
		"gpu_model":     in.GPUModel,
		"model_version": in.ModelVersion,
	}

	numericValues := map[string]int{
		"tokens_per_sample": in.TokensPerSample,
		"batch_size":        in.BatchSize,
	}

	for name, value := range stringValues {
		e.ConstitutivePropertyValues = append(e.ConstitutivePropertyValues, ConstitutivePropertyValue{
			ValueType: stringValueType,
			Value:     value,
			Property: TargetProperty{
				Identifier: name,
			},
		})
	}

	for name, value := range numericValues {
		e.ConstitutivePropertyValues = append(e.ConstitutivePropertyValues, ConstitutivePropertyValue{
			ValueType: numericValueType,
			Value:     value,
			Property: TargetProperty{
				Identifier: name,
			},
		})
	}

	payload, err := json.Marshal([]Entity{e})

	if err != nil {
		return "", err
	}

	// VV: Could also use /api/v0/ here
	requestURL := url + "/api/latest/actuators/custom_experiments/experiments/min_gpu_recommender/requests"

	req, err := http.NewRequest("POST", requestURL, bytes.NewBuffer(payload))
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("cannot send request with payload: %s due to %w", string(payload), err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)

	if err != nil {
		return "", fmt.Errorf("cannot read the POST response due to %w", err)
	}

	requestIDS := []string{}

	err = json.Unmarshal(body, &requestIDS)

	if err != nil {
		return "", fmt.Errorf("cannot unmarshal POST response body: %s due to %w", string(body), err)
	}

	if len(requestIDS) != 1 {
		return "", fmt.Errorf("expected exactly one request ID, got %d", len(requestIDS))
	}
	return requestIDS[0], nil

}

func RunPythonWrapperToCalcMinimumResourceRequirements(in MinGPURecommenderInput, pathToWrapperScript string, log logr.Logger) (ResourceRequirements, error) {
	timestamp := time.Now().UnixNano()
	pathOutput := os.TempDir() + "/" + fmt.Sprintf("min-gpu-recommender-%d", timestamp) + ".json"

	commands := []string{
		pathToWrapperScript,
		pathOutput,
		in.ModelName,
		in.Method,
		in.GPUModel,
		fmt.Sprintf("%d", in.TokensPerSample),
		fmt.Sprintf("%d", in.BatchSize),
		fmt.Sprintf("%s", in.ModelVersion),
	}
	log.Info("Running the wrapper script", "command", commands)

	cmd := exec.Command(
		"python", commands...,
	)

	cmd.Stderr = os.Stderr
	cmd.Stdout = os.Stdout

	err := cmd.Run()

	if err != nil {
		return ResourceRequirements{}, errors.Join(fmt.Errorf("error running python wrapper script"), err)
	}

	defer func() {
		// VV: Garbage collect the file
		os.Remove(pathOutput)
	}()

	raw, err := os.ReadFile(pathOutput)

	if err != nil {
		return ResourceRequirements{}, errors.Join(
			fmt.Errorf("error reading output of python wraper script from file %s", pathOutput),
			err,
		)
	}

	resources := ResourceRequirements{}
	err = json.Unmarshal(raw, &resources)

	if err != nil {
		return ResourceRequirements{}, errors.Join(
			fmt.Errorf("error decoding output of python wraper script from file %v", raw),
			err,
		)
	}

	if resources.Error != "" {
		return ResourceRequirements{}, fmt.Errorf("min_gpu_recommender identified the following error: %s", resources.Error)
	}

	return resources, nil
}

// Gets the commandline of a container.
// Assuming that the endpoint of the container image is a shell, there can be 2 ways to define the commandline of a container
// 1. `$shell -c $aStringWithPotentiallyManyCommands [optionally other arguments to $shell]`: Use a combination of container.Command and container.Args
// 2. `my actual command here`: Use a combination of container.Command and container.Args
// This function returns either (string, string, []) or a ("","", []string):
//
//	It returns (shellType, [all options other than $aStringWithPotentiallyManyCommands], $aStringWithPotentiallyManyCommands) when the container uses `$shell -c $aStringWithPotentiallyManyCommands [optionally other arguments to $shell]`.
//	It returns ("", append(container.Command, container.Args..), "") when the container uses a combination of container.Command and container.Args to define a command that is not of the form `$shell -c "a bunch of commands here"`.
func GetCommandLine(parts []string) (string, []string, string) {
	if len(parts) > 2 {
		if (path.Base(parts[0]) == "sh") || (path.Base(parts[0]) == "bash") {
			allOtherArgs := make([]string, 0, len(parts)-2)
			command := ""

			i := 1
			for i < len(parts) {
				m := parts[i]

				if m == "-c" {
					if i+1 < len(parts) {
						command = parts[i+1]
						i += 2
						continue
					}
				} else {
					allOtherArgs = append(allOtherArgs, m)
				}

				i++
			}

			return parts[0], allOtherArgs, command
		}
	}

	return "", parts, ""
}

func GetValueFromStringOrEnvVar(value string, envs []v1.EnvVar) string {
	value = strings.Trim(value, "\"")

	if strings.HasPrefix(value, "$") {
		value = strings.TrimPrefix(value, "$")
		value = strings.TrimPrefix(value, "{")
		value = strings.TrimSuffix(value, "}")

		for _, env := range envs {
			if env.Name == value && env.ValueFrom == nil {
				return env.Value
			}
		}

		// VV: TODO Is there a reason to complicate this by introducing errors here ?
		return ""
	} else {
		if strings.HasPrefix(value, "'") {
			value = strings.Trim(value, "'")
		}

		return value
	}
}

func ParseCommandLineString(cmdline string) map[string]string {
	config := map[string]string{}
	reader := strings.NewReader(cmdline)
	parser := syntax.NewParser(syntax.Variant(syntax.LangBash))

	graph, err := parser.Parse(reader, "commandline")

	if err != nil {
		return config
	}

	// VV: Recursively walk the graph and patch the cmdline of `accelerate launch` "CallExpr" nodes
	syntax.Walk(graph, func(n syntax.Node) bool {
		// VV: Return `false` when we don't want to go any deeper into the tree
		// This is for cases in which we know that we are either not going to find any more accelerate launch commands
		// Or we've already encountered a problem
		call, ok := n.(*syntax.CallExpr)

		if !ok {
			return true
		}

		if !isAccelerateLaunch(call) {
			return false
		}

		argvTokens, err := WordsToQuotedArgs(call.Args)
		if err != nil {
			return false
		}

		maps.Copy(config, ExtractCommandLineOptions(argvTokens, []string{
			"--per_device_train_batch_size",
			"--num_processes",
			"--peft_method",
			"--model_name_or_path",
			"--max_seq_length",
		}))

		return true
	})

	return config
}

// Rewrites the information collected from commandline options of accelerate launch into AutoConf fields using the values of environment variables available to accelerate launch
func ConvertCommandlineOptionsToAutoConfFields(config map[string]string, env []v1.EnvVar) {
	// VV: Figure out the value of perDeviceBatchSize and numProcesses, and set AUTOCONF_BATCH_SIZE to their product
	// then remove the 2 flags from the config map
	if perDeviceBatchSize, ok := config["--per_device_train_batch_size"]; ok {
		if numProcesses, ok := config["--num_processes"]; ok {

			perDeviceBatchSize = GetValueFromStringOrEnvVar(perDeviceBatchSize, env)

			perDeviceBatchSize, err := strconv.Atoi(perDeviceBatchSize)
			if err == nil {

				numProcesses = GetValueFromStringOrEnvVar(numProcesses, env)

				numProcesses, err := strconv.Atoi(numProcesses)

				if err == nil {
					config["AUTOCONF_BATCH_SIZE"] = fmt.Sprintf("%d", numProcesses*perDeviceBatchSize)
				}
			}
		}
	}

	delete(config, "--per_device_train_batch_size")
	delete(config, "--num_processes")

	if tokensPerSample, ok := config["--max_seq_length"]; ok {
		tokensPerSample := GetValueFromStringOrEnvVar(tokensPerSample, env)

		_, err := strconv.Atoi(tokensPerSample)

		if err == nil {
			config["AUTOCONF_TOKENS_PER_SAMPLE"] = tokensPerSample
		}
	}
	delete(config, "--max_seq_length")

	if peftMethod, ok := config["--peft_method"]; ok {
		peftMethod := GetValueFromStringOrEnvVar(peftMethod, env)

		if peftMethod == "lora" {
			config["AUTOCONF_METHOD"] = "lora"
		}
	} else {
		config["AUTOCONF_METHOD"] = "full"
	}

	delete(config, "--peft_method")

	if modelNameOrPath, ok := config["--model_name_or_path"]; ok {
		modelNameOrPath := GetValueFromStringOrEnvVar(modelNameOrPath, env)

		modelMap := map[string]string{
			"ibm-granite/granite-8b-code-base-4k":       "granite-8b-code-base",
			"ibm-granite/granite-3.0-8b-base":           "granite-3-8b",
			"ibm-granite/granite-3.1-2b-base":           "granite-3.1-2b",
			"ibm-granite/granite-3.1-3b-a800m-instruct": "granite-3.1-3b-a800m-instruct",
			"ibm-granite/granite-3.1-8b-instruct":       "granite-3.1-8b-instruct",
			"ibm-granite/granite-3b-code-base-128k":     "granite-3b-code-base-128k",
			"ibm-granite/granite-7b-base":               "granite-7b-base",
			"ibm-granite/granite-8b-code-base-128k":     "granite-8b-code-base-128k",
			"ibm-granite/granite-vision-3.2-2b":         "granite-vision-3.2-2b",
			"HuggingFaceTB/SmolLM2-135M":                "smollm2-135m",
			"ibm-granite/granite-4.0-micro":             "granite-4.0-micro",
			"ibm-granite/granite-4.0-h-1b":              "granite-4.0-h-1b",
			"ibm-granite/granite-4.0-350m":              "granite-4.0-350m",
			"ibm-granite/granite-4.0-h-small":           "granite-4.0-h-small",
			"ibm-granite/granite-4.0-1b":                "granite-4.0-1b",
			"ibm-granite/granite-4.0-h-micro":           "granite-4.0-h-micro",
			"ibm-granite/granite-4.0-h-tiny":            "granite-4.0-h-tiny",
			"meta-llama/Llama-2-7b":                     "llama-7b",
			"meta-llama/Llama-2-13b":                    "llama-13b",
			"meta-llama/Llama-2-70b":                    "llama2-70b",
			"meta-llama/Llama-3.1-8b":                   "llama3.1-8b",
			"meta-llama/Llama-3.1-70b":                  "llama3.1-70b",
			"meta-llama/Llama-3.1-405B-Instruct":        "llama3.1-405b",
			"mistralai/Mistral-7B-v0.1":                 "mistral-7b-v0.1",
			"mistralai/Mixtral-8x7B-Instruct-v0.1":      "mixtral-8x7b-instruct-v0.1",
		}

		modelNameOrPath = strings.TrimRight(modelNameOrPath, "/")

		if modelName, ok := modelMap[modelNameOrPath]; ok {
			config["AUTOCONF_MODEL_NAME"] = modelName
		} else {

			suffixes := map[string]string{
				"/LLaMa/models/hf/7B":                        "llama-7b",
				"/LLaMa/models/hf/13B":                       "llama-13b",
				"/LLaMa/models/hf/70B":                       "llama2-70b",
				"/LLaMa/models/hf/llama3-8b":                 "llama3-8b",
				"/LLaMa/models/hf/llama3.1-8b":               "llama3-1.8b",
				"/LLaMa/models/hf/llama3.1-70b":              "llama3.1-70b",
				"/LLaMa/models/hf/llama3.1-405b":             "llama3.1-405b",
				"/mistral-large/fp16_240620":                 "mistral-123b-v2",
				"/mistralai-mistral-7b-v0.1":                 "mistral-7b-v0.1",
				"/Mixtral-8x7B-Instruct-v0.1":                "mixtral-8x7b-instruct-v0.1",
				"/allam-1-13b-instruct-20240607":             "allam-1-13b",
				"/granite-13b-base-v2/step_300000_ckpt":      "granite-13b-v2",
				"/granite-20b-code-base-v2/step_280000_ckpt": "granite-20b-v2",
				"/granite-34b-code-base":                     "granite-34b-code-base",
			}

			for suffix, modelName := range suffixes {
				if strings.HasSuffix(modelName, suffix) {
					config["AUTOCONF_MODEL_NAME"] = modelName
				}
			}

		}
	}

	delete(config, "--model_name_or_path")
}

// Parse the commandline in containers[0] and its environment variables to extract information relevant to MinGPURecommenderInput
func ExtractPartialInformationFromCommandlineAndEnvVars(job *kubeflowv1.PyTorchJob) (map[string]string, error) {
	config := map[string]string{}

	// VV: No need for error checking here, we've already ensured that this container exists
	container := job.Spec.PyTorchReplicaSpecs[PrimaryPyTorchReplica].Template.Spec.Containers[0]
	_, allOtherargs, cmdline := GetCommandLine(append(container.Command, container.Args...))

	if cmdline != "" {
		maps.Copy(config, ParseCommandLineString(cmdline))
	} else {
		maps.Copy(config, ExtractCommandLineOptions(allOtherargs, []string{
			"--per_device_train_batch_size",
			"--num_processes",
			"--peft_method",
			"--model_name_or_path",
			"--max_seq_length",
		}))
	}

	ConvertCommandlineOptionsToAutoConfFields(config, container.Env)

	return config, nil
}

func ExtractMinGPURecommenderInput(job *kubeflowv1.PyTorchJob, defaultGPUModel, modelVersion string, log logr.Logger) (MinGPURecommenderInput, error) {
	primary := job.Spec.PyTorchReplicaSpecs[PrimaryPyTorchReplica]

	ret := MinGPURecommenderInput{}

	envVarNames := []string{"AUTOCONF_MODEL_NAME", "AUTOCONF_METHOD", "AUTOCONF_GPU_MODEL", "AUTOCONF_TOKENS_PER_SAMPLE", "AUTOCONF_BATCH_SIZE"}

	// VV: First, try extracting the information by parsing the cmdline of the job

	config, err := ExtractPartialInformationFromCommandlineAndEnvVars(job)

	if err != nil {
		log.Info("Unable to extract information from the commandline, falling back to environment variables", "error", err)
	}

	// VV: Next, see if they've exported any of the env-vars we're looking for

	for _, v := range primary.Template.Spec.Containers[0].Env {
		if slices.Contains(envVarNames, v.Name) {
			config[v.Name] = v.Value
		}
	}

	if defaultGPUModel != "" {
		config["AUTOCONF_GPU_MODEL"] = defaultGPUModel
	}

	if len(config) != len(envVarNames) {
		missing := []string{}

		for _, k := range envVarNames {
			if _, exists := config[k]; !exists {
				missing = append(missing, k)
			}
		}

		return ret, fmt.Errorf("missing required environment variables in primary PytorchReplicaSpec %+v", missing)
	}

	ret.Method = config["AUTOCONF_METHOD"]
	ret.ModelName = config["AUTOCONF_MODEL_NAME"]
	ret.GPUModel = config["AUTOCONF_GPU_MODEL"]
	ret.ModelVersion = modelVersion

	t, err := strconv.Atoi(config["AUTOCONF_TOKENS_PER_SAMPLE"])

	if err != nil {
		return ret, fmt.Errorf("error converting AUTOCONF_TOKENS_PER_SAMPLE to int: %w", err)
	}
	ret.TokensPerSample = t

	t, err = strconv.Atoi(config["AUTOCONF_BATCH_SIZE"])
	if err != nil {
		return ret, fmt.Errorf("error converting AUTOCONF_BATCH_SIZE to int: %w", err)
	}
	ret.BatchSize = t

	return ret, nil
}

func handleUpdateWrapperError(err error, log logr.Logger) (ctrl.Result, error) {
	message := fmt.Sprintf("Error updating AppWrapper: %s", err)

	if strings.Contains(message, "the object has been modified") {
		log.Info("Object has been modified externally - will requeue")
		log.V(4).Info("unable to update Object", "err", err)
		return ctrl.Result{Requeue: true}, nil
	} else {
		log.Error(err, "unable to update Object will requeue")
		return ctrl.Result{Requeue: true}, err
	}
}
