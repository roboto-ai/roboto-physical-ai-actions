# Development Guide

This guide provides detailed information for developing Roboto Actions using this template.

## Table of Contents

- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Action Parameters](#action-parameters)
- [Input Data](#input-data)
- [Output Data](#output-data)
- [Invoking Locally](#invoking-locally)
- [Build and Deployment](#build-and-deployment)

## Project Structure

#### `main.py`

The primary orchestrator of your action. This is where your action's core logic lives.

Simple actions may be defined in this single file, while complex actions benefit from being broken into separate modules, then imported and orchestrated here.

**Expected signature**:
```python
def main(context: roboto.InvocationContext) -> None:
    # Your action logic here
    ...
```

The function receives one argument: `context: roboto.action_runtime.InvocationContext`.

`InvocationContext` provides access to parameters, input data, working directories, organization/dataset information, and runtime configuration. Use `context.is_dry_run` to check if the action is running in dry-run mode. Use `context.log_level` to access the logging verbosity level (e.g., `logging.INFO`, `logging.DEBUG`).

#### `action.json`

Defines action configuration including:
- Action name and description
- Compute requirements (vCPU, memory, storage)
- Action parameters (see [Action Parameters](#action-parameters))
- Whether the action requires input data to be pre-downloaded into the working directory

See [ActionConfig](https://docs.roboto.ai/reference/python-sdk/roboto/domain/actions/index.html#roboto.domain.actions.ActionConfig) for details, as well as relevant documentation on https://docs.roboto.ai.

#### `Dockerfile`

Defines the Docker container image that runs your action. Modify this file to:
- Install system dependencies (see [System Dependencies](#system-dependencies))
- Customize the runtime environment

#### `.python-version`

Specifies the Python version for this action. This file is used by [pyenv](https://github.com/pyenv/pyenv) and similar tools to automatically set the correct Python version when working in this repository.

Update this file if you need a different Python version. Prefer a version compatible with or more recent than what's specified at repository initialization.

**Important:** Keep this version in sync with the Python version specified in [`Dockerfile`](Dockerfile).

## Development Workflow

### Adding Dependencies

#### Runtime Dependencies

Add third-party Python libraries required at runtime by your action to the `dependencies` array in the `[project]` section of [`pyproject.toml`](pyproject.toml).

Example:
```toml
[project]
dependencies = [
    "roboto"
]
```

After adding dependencies, rebuild your virtual environment:
```bash
$ ./scripts/setup.sh
```

#### Development Dependencies

Add Python-based tools used only as part of development and verification (such as linting and testing) to the `dev` array in the `[project.optional-dependencies]` section of [`pyproject.toml`](pyproject.toml).

Example:
```toml
[project.optional-dependencies]
dev = [
    "pytest",
    "ruff",
]
```

These dependencies are installed in your local virtual environment but are not included in the Docker image deployed to Roboto.

#### Troubleshooting: 'No solution found when resolving dependencies'

This project uses [uv](https://docs.astral.sh/uv/) to install dependencies, both locally and in the Docker image used to run the action on Roboto's hosted compute.

If you get an error from `./scripts/setup.sh` like:
```bash
$ ./scripts/setup.sh
Collecting uv
  Using cached uv-0.9.5-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (11 kB)
Using cached uv-0.9.5-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (21.3 MB)
Installing collected packages: uv
Successfully installed uv-0.9.5
  × No solution found when resolving dependencies:
  ╰─▶ Because only library<=1.2.3 is available and roboto-to-lerobot depends on library>=2.3.4, we can conclude that your requirements are unsatisfiable.
```

Try clearing the `uv` cache and re-running `./scripts/setup.sh`:
```bash
$ .venv/bin/uv cache clean
$ ./scripts/setup.sh
```

If the problem persists, confirm that the version of the library you specified is published on [PyPI](https://pypi.org/).

#### System Dependencies

While this action is written in Python, it runs within a Docker container defined in [`Dockerfile`](Dockerfile).

Some actions or third-party libraries require system dependencies that may be installed on your workstation but are not available in the container image.

To install additional system dependencies, add `RUN apt-get install` instructions in the `Dockerfile`. See the commented section beginning with `# -- INSTALL SYSTEM DEPENDENCIES --`.

Example:
```dockerfile
# -- INSTALL SYSTEM DEPENDENCIES --
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*
```

**Important**: Always test the container's runtime environment before deploying to Roboto by building the image and invoking it locally (see [Invoking Locally](#invoking-locally)).

### Running Tests and Linting

To verify your code quality and run tests, use the provided verification script:

```bash
$ ./scripts/verify.sh
```

This script runs:
- **Linting**: Uses [`ruff`](https://docs.astral.sh/ruff/) to check code style and quality
- **Testing**: Runs the test suite with [`pytest`](https://docs.pytest.org/en/stable/)

The script will exit with an error if any checks fail. Run this before committing changes or as part of CI.

`ruff` provides other useful code quality utilities:
- **Auto-fix linting errors**: `.venv/bin/ruff check --fix`
- **Auto-format**: `.venv/bin/ruff format`

### Code Organization Best Practices

- **Single entry point**: Export a single `main()` function with the expected signature
- **Keep `main.py` focused**: It should orchestrate your action's workflow, not contain all implementation details
- **Use modules for complex logic**: Break complex actions into separate modules under `src/roboto_to_lerobot/`
- **Type hints**: Use type hints for better code clarity and IDE support
- **Logging**: Use `context.log_level` to access the logging verbosity level. Use it to set the logging level in your action code via `logger.setLevel(context.log_level)`.
- **Dry-run support**: Use `context.is_dry_run` to enable local invocation without side effects

**Dry-run mode**: When `context.is_dry_run` is `True`, consider gating operations that have side effects, such as:
- Uploading files to Roboto datasets
- Modifying metadata
- Making external API calls that are not idempotent

## Action Parameters

### Overview

A Roboto Action can be compared to a function. Like functions, actions can define required or optional arguments. Actions refer to these arguments as **parameters**. Parameters can be set when the action is invoked, either manually or by a Roboto Trigger.

### Defining Parameters

Parameters are defined in the `parameters` array within `action.json`. Each parameter follows the [ActionParameter](https://docs.roboto.ai/reference/python-sdk/roboto/domain/actions/action_record/index.html#roboto.domain.actions.action_record.ActionParameter) model schema.

Properties:
- `name`: Identifier for the parameter. Use this at runtime to lookup the parameter value.
- `description`: Human-readable description of the parameter's purpose
- `required`: Whether the parameter must be provided (defaults to `false`)
- `default`: Default value if not provided (only for optional parameters)

Example:
```json
{
    "name": "my-action",
    "description": "My action description",
    "parameters": [
        {
            "name": "threshold",
            "description": "Detection threshold",
            "required": true
        },
        {
            "name": "labeling_service_api_key",
            "description": "API key for the labeling service",
            "required": true
        },
        {
            "name": "output_format",
            "description": "Output format (json or csv)",
            "default": "json"
        }
    ]
}
```

### Passing Secrets Through Parameters

Roboto supports passing secrets to actions that have been registered in Roboto's secrets store.

Parameters that receive secret values are defined in `action.json` like any other parameter. When invoking the action, format the value as `roboto-secret://<secret-name>`. At runtime, use [`InvocationContext.get_secret_parameter()`](https://docs.roboto.ai/reference/python-sdk/roboto/action_runtime/invocation_context/index.html#roboto.action_runtime.invocation_context.InvocationContext.get_secret_parameter) to resolve the secret to its actual value.

Example:
```bash
$ roboto secrets write my-labeling-service-api-key API_KEY_VALUE
$ roboto actions invoke-local --parameter labeling_service_api_key="roboto-secret://my-labeling-service-api-key"
```

### Using Parameters in Your Action

Action parameters are passed as environment variables in both hosted compute and local Docker execution.

**Local invocation**: When you run `roboto actions invoke-local --parameter threshold=60 --parameter labeling_service_api_key="roboto-secret://my-labeling-service-api-key"`, the Roboto CLI converts your CLI arguments into environment variables before launching the Docker container.

**Hosted compute**: When run by Roboto on its platform, the platform infrastructure sets the same environment variables.

The Roboto Python SDK's [`roboto.InvocationContext`](https://docs.roboto.ai/reference/python-sdk/roboto/action_runtime/invocation_context/index.html#roboto.action_runtime.invocation_context.InvocationContext) provides convenient access to these parameters. See:

- [get_parameter()](https://docs.roboto.ai/reference/python-sdk/roboto/action_runtime/invocation_context/index.html#roboto.action_runtime.invocation_context.InvocationContext.get_parameter)

- [get_optional_parameter()](https://docs.roboto.ai/reference/python-sdk/roboto/action_runtime/invocation_context/index.html#roboto.action_runtime.invocation_context.InvocationContext.get_optional_parameter)

- [get_secret_parameter()](https://docs.roboto.ai/reference/python-sdk/roboto/action_runtime/invocation_context/index.html#roboto.action_runtime.invocation_context.InvocationContext.get_secret_parameter)

Example usage in `main.py`:
```python
def main(context: roboto.InvocationContext) -> None:
    threshold = context.get_parameter("threshold")
    output_format = context.get_optional_parameter("output_format", "json")
    labeling_service_api_key = context.get_secret_parameter("labeling_service_api_key")
    ...
```

### Parameter Type Handling

All parameter values are received as strings, regardless of the intended type. Your action code is responsible for parsing and validating parameter values.

**Parsing different types**:
```python
import json

def main(context: roboto.InvocationContext) -> None:
    # Integer parameter
    threshold = int(context.get_parameter("threshold"))

    # Float parameter
    confidence = float(context.get_parameter("confidence"))

    # Boolean parameter
    enable_feature = context.get_parameter("enable_feature").lower() == "true"

    # JSON/complex parameter
    config = json.loads(context.get_parameter("config"))

    # List parameter (comma-separated)
    tags = context.get_parameter("tags").split(",")
```

**Best practice**: Always validate and handle parsing errors to provide clear error messages:
```python
def main(context: roboto.InvocationContext) -> None:
    threshold = int(context.get_parameter("threshold"))
    if threshold < 0 or threshold > 100:
        raise ValueError(
            "Invalid threshold parameter: threshold must be between 0 and 100"
        )
```

### Accessing Runtime Information: InvocationContext vs Environment Variables

**Recommended approach**: Always use `InvocationContext` methods to access runtime information such as parameters, input data, and working directories.

While the Roboto platform sets various `ROBOTO_*` environment variables, these are implementation details that may change between versions. Avoid relying on them directly in your action code.

**Example of recommended usage**:
```python
import os
import roboto

def main(context: roboto.InvocationContext) -> None:
    # ✅ Recommended: Use InvocationContext methods
    threshold = context.get_parameter("threshold")
    action_input = context.get_input()
    dataset_id = context.dataset_id

    # ❌ Avoid: Accessing other ROBOTO_* environment variables directly
    # param = os.environ["ROBOTO_PARAM_THRESHOLD"]  # Don't do this
```

## Input Data

### Overview

Most actions operate on data uploaded to Roboto, and many upload results back to Roboto. Managing input data to seamlessly enable both local development and production usage is critical.

Roboto provides two approaches for working with input data:

### Approach A: Data Specified at Invocation Time

In this approach, input data is specified when the action is invoked using a RoboQL query or a glob pattern that matches one or more files from a specified dataset.

When an action is automatically triggered by Roboto, the trigger specifies the input data for the invocation.

**When to use:** This is the most common approach, and is suggested for use with actions invoked by both event-based triggers (e.g., file upload or ingestion) and scheduled triggers.

**How to use:** The Roboto Python SDK's [`roboto.InvocationContext`](https://docs.roboto.ai/reference/python-sdk/roboto/action_runtime/invocation_context/index.html#roboto.action_runtime.invocation_context.InvocationContext) provides helpers for accessing this data. See [get_input()](https://docs.roboto.ai/reference/python-sdk/roboto/action_runtime/invocation_context/index.html#roboto.action_runtime.invocation_context.InvocationContext.get_input) and [ActionInput](https://docs.roboto.ai/reference/python-sdk/roboto/action_runtime/action_input/index.html#roboto.action_runtime.action_input.ActionInput) for more.


**Example: Processing Files**
```python
import roboto
import json

def main(context: roboto.InvocationContext) -> None:
    action_input = context.get_input()

    # Iterate through input files
    for file, local_path in action_input.files:
        print(f"Processing file: {file.relative_path}")
        print(f"  File ID: {file.file_id}")
        print(f"  Local path: {local_path}")
```


**Example local invocation**:

```bash
# Query for files using RoboQL
$ roboto --log-level=info actions invoke-local --file-query "dataset_id=ds_123 AND path LIKE '%.log'"

# Specify files by dataset ID and paths
$ roboto --log-level=info actions invoke-local --dataset ds_123 --file-path "logs/file1.log" --file-path "logs/file2.log"
```


#### Using Queries to Specify Input Data

When using `--topic-query` or `--file-query`, provide RoboQL to match data stored in Roboto. The query syntax allows you to filter resources based on their attributes, as well as based on their relationships with other resources.

Refer to [RoboQL documentation](https://docs.roboto.ai/roboql/index.html) for the definitive guide, and use the Roboto web application's Search UI to test queries and inspect result sets.

**Important**: Queries match files already uploaded to Roboto and topics already ingested from those files, not local filesystem files.

#### Configuring Input Data Download

The `requires_downloaded_inputs` field in [`action.json`](action.json) controls whether input data are automatically downloaded before your action runs.

**Note:** This argument only has an effect if input is specified as a file query or a dataset_id/file paths combination.

**`requires_downloaded_inputs: true`** (Default)
- Input files are downloaded to `ROBOTO_INPUT_DIR` ([`InvocationContext::input_dir`](https://docs.roboto.ai/reference/python-sdk/roboto/action_runtime/invocation_context/index.html#roboto.action_runtime.invocation_context.InvocationContext.input_dir)) before the action executes.
- Files are accessible via the filesystem at paths provided by `context.get_input()`.

**`requires_downloaded_inputs: false`**
Only file metadata ([instances of `roboto.File`](https://docs.roboto.ai/reference/python-sdk/roboto/domain/files/file/index.html#roboto.domain.files.file.File)) is available via `context.get_input().files`. `local_path` will always equal `None` in tuples in that collection.

### Approach B: Query for Data at Runtime

In this approach, action code queries Roboto for data at runtime, enabling it to operate on any data queryable from the Roboto API.


**Example: Query for Files at Runtime**
```python
import roboto

def main(context: roboto.InvocationContext) -> None:
    roboto_search = roboto.RobotoSearch.for_roboto_client(context.roboto_client)

    # Find files matching a query
    for file in roboto_search.find_files("path LIKE '%.log' AND size < 1000000"):
        print(f"Found file: {file.relative_path}")

        # Download the file if needed
        # local_path = context.input_dir / file.relative_path
        # file.download(local_path)
        # ...Process file content...
```


**When to use**: This approach is most common for actions performing batch processing that will be invoked manually, with no changes expected in the query used to gather data. Prefer using [Approach A](#approach-a-input-data-specified-at-invocation-time), as it allows you to more flexibly define how input data is selected, including the use of `LIMIT` when testing.

## Output Data

### Overview

Actions often need to write output files, such as analysis results, processed data, or generated reports. Understanding how output files are handled in different execution environments is important for developing and testing your action.

### Writing Output Files

Write output files to the directory specified by [`InvocationContext.output_dir`](https://docs.roboto.ai/reference/python-sdk/roboto/action_runtime/invocation_context/index.html#roboto.action_runtime.invocation_context.InvocationContext.output_dir). This directory is available in both local and hosted execution environments.

**Example**:
```python
import roboto

def main(context: roboto.InvocationContext) -> None:
    result = context.output_dir / "results.json"
    result.write_text("Hello, world!")
```

### Automatic Upload Behavior

**Hosted Platform**: Files written to the output directory are automatically uploaded after successful completion. The upload destination is determined as follows:
- If an output dataset is specified at invocation time, files are uploaded there
- If no output dataset is specified, the platform attempts to infer it from input data:
  - If all input data belongs to a single dataset, that dataset is used
  - If input data spans multiple datasets, automatic upload is skipped

**Local Invocation**: When invoked locally, files written to the output directory (typically `.workspace/output/`) are **not** automatically uploaded. You can inspect them locally for testing and debugging purposes.

### Advanced Output Control

For more control over output uploads, such as uploading files to specific locations or with custom metadata, use the Roboto Python SDK directly. See the [Roboto Python SDK documentation](https://docs.roboto.ai/reference/python-sdk.html) for up-to-date documentation.

## Invoking Locally

### Overview

Local invocation always runs in a Docker container to ensure production parity. The Docker image is rebuilt on each invocation (Docker's layer caching makes this fast when nothing has changed).

### Basic Usage

Use the `roboto actions invoke-local` command to run your action locally:

```bash
# See all available options
$ roboto actions invoke-local --help

# Specify topics to use as input and parameters defined in action.json
$ roboto --log-level=info actions invoke-local --topic-query="msgpaths[cpuload.load].max > 0.9" --parameter threshold=60
```

See the [Local Action Invocation section in README.md](README.md#local-action-invocation) for complete usage documentation and examples.

### How It Works

When you run `roboto actions invoke-local`:

1. **Build**: The Docker image defined by [Dockerfile](Dockerfile) is built (or rebuilt if source or dependencies changed)
2. **Prepare**: Workspace directories are readied on your host machine
3. **Download**: Input data is downloaded if specified (`requires_downloaded_inputs` in `action.json`)
4. **Launch**: A Docker container is launched with:
   - Your workspace mounted at its full host path (e.g., `/home/user/project/.workspace`)
   - All necessary environment variables set
   - Your user/group ID to ensure proper file permissions
5. **Execute**: The action executes inside the container

### Workspace

Local invocation creates a `.workspace/` directory in the root of the action repo for temporary files, input data, and output. This directory is used only for local development and should not be committed to version control. It is cleared between invocations unless you use the `--preserve-workspace` flag.

## Build and Deployment

### Building the Docker Image

```bash
$ ./scripts/build.sh
```

Builds the Docker image locally. The image is tagged with your action name.

### Deploying to Roboto Platform

```bash
$ ./scripts/deploy.sh
```

This script:
1. Builds the Docker image
2. Pushes it to Roboto's container registry
3. Creates or updates the action definition on the Roboto platform using configuration from [`action.json`](action.json)

**Prerequisites**: You must have a personal access token accessible in your environment. See [Programmatic Access](https://docs.roboto.ai/getting-started/programmatic-access.html#personal-access-token) for more.

**Note**: If you belong to multiple Roboto organizations, specify the target organization by setting the `ROBOTO_ORG_ID` environment variable or passing the organization ID as the first argument to `deploy.sh`.
