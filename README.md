# Automatic License Plate Recognition System (ALPR)

### Luxonis OAK-D CM4 + DepthAI Pipeline

## Overview

This project implements a complete **Automatic License Plate Recognition
(ALPR)** system deployed on the Luxonis OAK-D CM4 using the DepthAI SDK.
The system is designed for real-time vehicle monitoring, detection,
tracking, and structured event logging.

It combines yolo object detection with on-device
processing to achieve an efficient, standalone pipeline suitable for
edge deployment scenarios such as traffic monitoring, access control,
and smart surveillance systems.

The system performs:

-   Vehicle detection
-   License plate detection
-   Object tracking with persistent IDs
-   Line-crossing event detection
-   Structured logging of detected vehicles and plates
-   Real-time visualization

## Features

### Detection Pipeline

-   Real-time vehicle detection using YOLOv6n 
-   Dedicated license plate detection model using YOLOv8n
-   Inference executed directly on-device

### Tracking

-   Multi-object tracking with consistent ID assignment
-   Per-object state management across frames
-   Robust handling of occlusions and re-identification

### Line-Crossing Detection

-   Configurable virtual line
-   Detection of crossing direction
-   Event triggering only on valid transitions

### License Plate Recognition

-   Extraction and processing of detected plates
-   Support for variable-length plate formats
-   Standardized formatting and logging

### Logging System

-   Structured event logs for each detected vehicle
-   Includes:
    -   Timestamp
    -   Object ID
    -   Plate number (formatted)
    -   Crossing event

### Visualization

-   Bounding boxes for vehicles and plates
-   Line-crossing indicators

## System Architecture

The pipeline is built using the node-based architecture of the
DepthAI SDK.

### Core Components

1.  Video Input
2.  Vehicle Detection Node
3.  License Plate Detection Node
4.  Tracking Module
5.  Event Logic Layer
6.  Recognition & Formatting
7.  Logging Module
8.  Visualization Layer

## Repository Structure

ALPR_System/ │ ├── main.py ├── models/  ├── logs/ └──
README.md

## Requirements

### Hardware

-   Luxonis OAK-D CM4

### Software

-   Python 3.x
-   DepthAI SDK
-   OpenCV
-   NumPy

## Usage

Run the main pipeline:

python main.py

## Output

### Visual Output

-   Annotated video stream

### Logs

Timestamp \| Object ID \| Plate Number \| Event

Example: 2025-07-25 14:32:10 \| ID: 07 \| Plate: AB123CD \| Crossing: IN

## Design Considerations

-   Edge-first processing
-   Modular architecture
-   Robust tracking
-   Scalability

## Future Improvements

-   Multi-camera support
-   Cloud integration
-   Speed estimation

## Author

Developed as part of Master thesis "Vehicle and pedestraian recognition in restricted access areas using edge devices"
