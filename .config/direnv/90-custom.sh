#!/bin/bash

GIT_ROOT="$(git rev-parse --show-toplevel)"

dotenv_if_exists "$GIT_ROOT/.env"
