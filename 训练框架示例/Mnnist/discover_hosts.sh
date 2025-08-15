#!/bin/bash

timestamp=$(date +%s)

if [ $timestamp -lt $((timestamp - 120)) ]; then
  echo "mpi-job-master-0:1"
  echo "mpi-job-worker-0:1"
else
  echo "mpi-job-worker-0:1"
fi