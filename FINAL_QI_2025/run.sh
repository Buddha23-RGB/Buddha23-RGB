#!/bin/bash
docker build -t FINAL_QI_2025 .
docker run -p 3000:3000 -d FINAL_QI_2025