# Causal Models

## Installation

1. Clone the repo
2. Create a virtual environment
3. Install the requirements: `pip install -r requirements`
4. Install the `causal_networks` locally in edit mode: `pip install -e .`


## Using Docker

A docker file is available which allows for iterative development and running
experiments. To build a new image and use it, follow the proceeding steps.

1. Create GitHub personal access token. Ideally use a fine-grained one which has access
   only to the contents of this repository.

2. Create a file named `.env` with the following contents

```bash
GITHUB_USER=
GITHUB_PAT=
GIT_NAME=""
GIT_EMAIL=""
SSH_PUBKEY=""
```

3. Fill in the details with your GitHub username, your GitHub PAT, your name as you'd
   like it to appear in git commit messages, the email you'd like to use for git commits
   and the SSH public key you'd like to use to access the container.

4. Build the image using the following command:

```
docker build -t DOCKER_REPO:DOCKER_TAG --secret id=my_env,src=.env .
```

replacing `DOCKER_REPO` and `DOCKER_TAG` with the appropriate details.

5. Push the image to the Docker Hub, ready for use.


## TODO

- Currently `TransformerVariableAlignment` compares the low-level and DAG outputs on all
  tokens, including pad tokens. This probably doesn't make any sense, and is messing up
  training and evaluation.
- The interchange intervention dataset specifies for each datapoint a subset of the
  nodes which are intervened on. Currently the way this is done is by replacing the
  values of the non-intervened nodes with the values taken on the base input. This works
  fine for nodes which don't lie above or below each other, but otherwise doesn't work.
  It would be good to allow only intervening on the selected nodes, leaving the rest as
  they are. Could be done with a mask on the nodes.
- When computing the activation values on the source inputs, there's no need to run the
  model beyond the last layer. It should be easy to stop execution at this point by
  raising an exception in a hook. This should make precomputing faster.