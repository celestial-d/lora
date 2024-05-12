#!/bin/bash

perf record -F 5 -a -g $(pidof server.py)