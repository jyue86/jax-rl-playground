# jax-rl-playground

## Setup
### Local
Install [uv](https://docs.astral.sh/uv/) before the following.

```bash
uv venv --python 3.11 
uv sync
```
### Docker
Build the docker and run the container
```bash
docker build -f Dockerfile -t cisl/jax-rl:1.0 .
docker run --rm -it -v $(pwd):/app cisl/jax-rl:1.0 bash  
```