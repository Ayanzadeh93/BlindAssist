#!/usr/bin/env python3
"""Command-line argument parsing module for AIDGPT application."""
import argparse


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AIDGPT application for time-based video processing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument('--input', type=str, default='0',
                             help='Input source (0 for webcam, or path for video file)')
    input_group.add_argument('--phone_camera', type=str, default=None,
                             help='Phone camera IP/URL (priority over --input)')
    input_group.add_argument('--video_file', type=str, default=None,
                             help='Path to video file for processing')

    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output', type=str, default=None,
                              help='Output video path (optional)')

    ai_group = parser.add_argument_group('AI Model Options')
    ai_group.add_argument('--ai_model', type=str,
                          choices=['gpt-4o', 'gpt-5', 'gpt-5-nano', 'claude-3.5', 'llama-3.2-vision'],
                          default='gpt-4o',
                          help='AI model to use')
    ai_group.add_argument('--gpt_cooldown', type=float, default=5.0,
                          help='[DEPRECATED] Minimum time between AI calls (use --inference_interval)')
    ai_group.add_argument('--inference_interval', type=float, default=5.0,
                          help='Time between inference cycles in seconds')
    ai_group.add_argument('--overlay_duration', type=float, default=2.0,
                          help='Duration to display inference results on screen in seconds')
    ai_group.add_argument('--batch_size', type=int, default=1,
                          help='Batch size for processing (unused)')
    ai_group.add_argument('--processing_interval', type=int, default=10,
                          help='[DEPRECATED] Not used. Inference is time-based.')

    processing_group = parser.add_argument_group('Processing Options')
    processing_group.add_argument('--keep_days', type=int, default=7,
                                  help='Number of days to keep processed data')
    processing_group.add_argument('--background_mode', action='store_true',
                                  help='Run in background mode without visualization')
    processing_group.add_argument('--visualize', action='store_true', default=True,
                                  help='Show visualization (default True)')
    processing_group.add_argument('--no-visualize', action='store_true',
                                  help='Disable visualization')

    logging_group = parser.add_argument_group('Logging Options')
    logging_group.add_argument('--log_csv', action='store_true',
                               help='Enable CSV logging of GPT responses')
    logging_group.add_argument('--csv_output', type=str, default='gpt_responses.csv',
                               help='CSV output file path')
    logging_group.add_argument('--log_interval', type=float, default=1.0,
                               help='Minimum interval between CSV log entries in seconds')

    args = parser.parse_args()
    if args.no_visualize:
        args.visualize = False
    if args.phone_camera and args.video_file:
        parser.error("Cannot specify both --phone_camera and --video_file. Choose one.")
    return args


def get_argument_summary(args: argparse.Namespace) -> str:
    summary_lines = ["Command-line Arguments:"]
    summary_lines.append(f"  Input: {args.input}")
    if args.phone_camera:
        summary_lines.append(f"  Phone Camera: {args.phone_camera}")
    if args.video_file:
        summary_lines.append(f"  Video File: {args.video_file}")
    if args.output:
        summary_lines.append(f"  Output: {args.output}")
    summary_lines.append(f"  AI Model: {args.ai_model}")
    summary_lines.append(f"  Inference Interval: {args.inference_interval}s")
    summary_lines.append(f"  Overlay Duration: {args.overlay_duration}s")
    summary_lines.append(f"  GPT Cooldown (deprecated): {args.gpt_cooldown}s")
    summary_lines.append(f"  Background Mode: {args.background_mode}")
    summary_lines.append(f"  Visualization: {args.visualize}")
    if args.log_csv:
        summary_lines.append(f"  CSV Logging: {args.csv_output} (interval: {args.log_interval}s)")
    return "\n".join(summary_lines)







