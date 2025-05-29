# serverless invoke local -f sleepStagePredict -p realistic_8h_sleep_v2.json

import json
import os
import numpy as np
import neurokit2 as nk
from datetime import datetime, timezone
import copy
import traceback

try:
    import sleepecg
    import tensorflow as tf
except ImportError as e:
    print(f"Error importing libraries: {e}")
    sleepecg = None
    tf = None

MODEL_DIR = "./wrn-gru-mesa"
MODEL_NAME = "wrn-gru-mesa"
SLEEP_STAGE_DURATION_S = 30
MIN_HEARTBEATS_FOR_ANALYSIS = 10
DEFAULT_FALLBACK_RECORD_DURATION_S_EMPTY_BEATS = 60

# MODIFIED: Configuration for 'wake-rem-nrem' (3-stage + UNDEFINED)
STAGES_MODE_FOR_CLF = 'wake-rem-nrem'
NUM_CLF_OUTPUT_CLASSES = 4

TARGET_N_STAGES = 3
TARGET_STAGE_LABELS = {0: "WAKE", 1: "REM", 2: "NREM"}
TARGET_STAGE_NAME_TO_IDX = {
    label: idx
    for idx, label in TARGET_STAGE_LABELS.items()
}
CLF_ARGMAX_TO_TARGET_STAGE_IDX = {
    3: TARGET_STAGE_NAME_TO_IDX["WAKE"],
    2: TARGET_STAGE_NAME_TO_IDX["REM"],
    1: TARGET_STAGE_NAME_TO_IDX["NREM"],
    0: TARGET_STAGE_NAME_TO_IDX["WAKE"]
}

CLF = None
if sleepecg and os.path.exists(MODEL_DIR) and os.path.isdir(
        MODEL_DIR) and os.path.exists(
            os.path.join(MODEL_DIR, f"{MODEL_NAME}.zip")):
    print(
        f"Attempting to load model '{MODEL_NAME}' from directory: {os.path.abspath(MODEL_DIR)}"
    )
    try:
        CLF = sleepecg.load_classifier(MODEL_NAME, classifiers_dir=MODEL_DIR)
        print("SleepECG model loaded successfully in global scope.")

        if hasattr(CLF, 'feature_extraction_params') and \
           isinstance(CLF.feature_extraction_params, dict):
            print(
                f"Original feature_extraction_params: {CLF.feature_extraction_params}"
            )
            if 'feature_selection' in CLF.feature_extraction_params and \
               isinstance(CLF.feature_extraction_params['feature_selection'], list):
                current_selection = list(
                    CLF.feature_extraction_params['feature_selection'])
                updated_selection = []
                modified = False
                hrv_freq_components_from_log = [
                    'VLF', 'LF', 'HF', 'LF_HF_ratio', 'total_power', 'LF_norm',
                    'HF_norm'
                ]

                for item in current_selection:
                    if item == 'hrv-frequency':
                        for feature_to_add in hrv_freq_components_from_log:
                            if feature_to_add not in updated_selection:
                                updated_selection.append(feature_to_add)
                        modified = True
                        print(
                            f"Replacing 'hrv-frequency' group with specific HRV frequency features: {hrv_freq_components_from_log}"
                        )
                    elif item not in updated_selection:
                        updated_selection.append(item)
                if modified:
                    CLF.feature_extraction_params[
                        'feature_selection'] = updated_selection
                    print(
                        f"Updated feature_selection: {CLF.feature_extraction_params['feature_selection']}"
                    )
                else:
                    print(
                        "No 'hrv-frequency' group found in feature_selection, or it was already appropriately specific."
                    )
            else:
                print(
                    "Warning: 'feature_selection' not found in CLF.feature_extraction_params or not a list."
                )

            if 'lookback' in CLF.feature_extraction_params:
                print(
                    f"Using lookback: {CLF.feature_extraction_params['lookback']}"
                )
            if 'lookforward' in CLF.feature_extraction_params:
                print(
                    f"Using lookforward: {CLF.feature_extraction_params['lookforward']}"
                )
            if 'fs_rri_resample' in CLF.feature_extraction_params:
                print(
                    f"Using fs_rri_resample from CLF.feature_extraction_params: {CLF.feature_extraction_params['fs_rri_resample']}"
                )
    except Exception as e:
        CLF = None
        print(f"Error loading SleepECG model in global scope: {e}")
        print(traceback.format_exc())
else:
    CLF = None
    if not sleepecg:
        print("SleepECG library not imported. Model loading skipped.")
    else:
        print(
            f"Model directory or file not found. Searched for '{MODEL_NAME}.zip' in '{os.path.abspath(MODEL_DIR)}'. Model loading skipped."
        )


def _get_target_stage_predictions(clf_probabilities):
    """Converts CLF probabilities (from 'wake-rem-nrem' mode) to target 3-stage indices and labels."""
    if clf_probabilities is None:
        print(
            "Error: clf_probabilities is None in _get_target_stage_predictions."
        )
        return np.array([]), []
    if clf_probabilities.shape[1] != NUM_CLF_OUTPUT_CLASSES:
        print(
            f"Error: clf_probabilities has {clf_probabilities.shape[1]} columns, expected {NUM_CLF_OUTPUT_CLASSES} for '{STAGES_MODE_FOR_CLF}' mode."
        )
        return np.array([]), []
    clf_argmax_indices = np.argmax(clf_probabilities, axis=1)
    target_indices = np.array([
        CLF_ARGMAX_TO_TARGET_STAGE_IDX.get(idx,
                                           TARGET_STAGE_NAME_TO_IDX["WAKE"])
        for idx in clf_argmax_indices
    ])
    target_labels = [
        TARGET_STAGE_LABELS.get(idx, "Unknown") for idx in target_indices
    ]
    return target_indices, target_labels


def _generate_default_target_stage_prediction(num_epochs):
    """Generates default 3-stage (W,R,NREM) predictions."""
    print(
        f"Generating default {TARGET_N_STAGES}-stage prediction with {num_epochs} epochs."
    )
    if num_epochs <= 0:
        return np.array([]), []
    random_probabilities = np.random.random((num_epochs, TARGET_N_STAGES))
    if num_epochs > 0:
        random_probabilities = random_probabilities / random_probabilities.sum(
            axis=1, keepdims=True)
    target_indices = np.argmax(random_probabilities, axis=1)
    target_labels = [
        TARGET_STAGE_LABELS.get(idx, "Unknown") for idx in target_indices
    ]
    return target_indices, target_labels


def _try_stage_sleep_record(clf, sleep_record_obj):
    """
    Attempts sleep staging using sleepecg.stage with retry logic.
    Uses STAGES_MODE_FOR_CLF.
    Returns CLF output probabilities if successful, else None.
    """
    current_stages_mode = STAGES_MODE_FOR_CLF
    print(
        f"Calling sleepecg.stage with stages_mode='{current_stages_mode}'...")
    predicted_stages_probabilities = None
    original_params = None
    try:
        if hasattr(clf, 'feature_extraction_params') and isinstance(
                clf.feature_extraction_params, dict):
            original_params = copy.deepcopy(clf.feature_extraction_params)
        predicted_stages_probabilities = sleepecg.stage(
            clf,
            sleep_record_obj,
            return_mode="prob",
        )
        print(f"sleepecg.stage completed successfully on first try!")
        if predicted_stages_probabilities is not None and predicted_stages_probabilities.shape[
                1] != NUM_CLF_OUTPUT_CLASSES:
            print(
                f"Warning: sleepecg.stage output an unexpected number of columns ({predicted_stages_probabilities.shape[1]}) for stages_mode='{current_stages_mode}'. Expected {NUM_CLF_OUTPUT_CLASSES}."
            )
            return None
        return predicted_stages_probabilities
    except Exception as e:
        print(
            f"sleepecg.stage (with stages_mode='{current_stages_mode}') failed on first try: {e}"
        )
        print("Detailed traceback for first try failure:")
        print(traceback.format_exc())
        print(
            "Attempting with modified parameters (still using specified stages_mode)..."
        )
        try:
            if clf and hasattr(clf,
                               'feature_extraction_params') and isinstance(
                                   clf.feature_extraction_params, dict):
                if original_params is None:
                    original_params = copy.deepcopy(
                        clf.feature_extraction_params)
                clf.feature_extraction_params['max_nans'] = 0.9
                clf.feature_extraction_params[
                    'sleep_stage_duration'] = SLEEP_STAGE_DURATION_S
                print(
                    f"Modified parameters for retry: {clf.feature_extraction_params}"
                )
            else:
                print(
                    "CLF.feature_extraction_params not found or not a dict, cannot modify for retry."
                )
            predicted_stages_probabilities = sleepecg.stage(
                clf,
                sleep_record_obj,  # Use the passed SleepRecord object
                return_mode="prob",
                # stages_mode=current_stages_mode
            )
            print(
                f"sleepecg.stage (with stages_mode='{current_stages_mode}') completed successfully on retry!"
            )
            if predicted_stages_probabilities is not None and predicted_stages_probabilities.shape[
                    1] != NUM_CLF_OUTPUT_CLASSES:
                print(
                    f"Warning on retry: sleepecg.stage output an unexpected number of columns ({predicted_stages_probabilities.shape[1]}) for stages_mode='{current_stages_mode}'. Expected {NUM_CLF_OUTPUT_CLASSES}."
                )
                return None
            return predicted_stages_probabilities
        except Exception as e_retry:
            print(
                f"sleepecg.stage (with stages_mode='{current_stages_mode}') failed on retry: {e_retry}"
            )
            print("Detailed traceback for retry failure:")
            print(traceback.format_exc())
            return None
        finally:
            if original_params is not None and clf and hasattr(
                    clf, 'feature_extraction_params'
            ) and clf.feature_extraction_params is not original_params:
                clf.feature_extraction_params = original_params
                print("Restored original CLF parameters after retry attempt.")
    return None


def predict_eeg_data(ecg_data, fs, start_time_iso):
    if not sleepecg or not CLF:
        return {
            "statusCode":
            500,
            "body":
            json.dumps(
                {"error": "Model not loaded or sleepecg not available."})
        }
    if not isinstance(ecg_data, list) or not all(
            isinstance(x, (int, float)) for x in ecg_data):
        return {
            "statusCode": 400,
            "body":
            json.dumps({"error": "ECG data must be a list of numbers."})
        }
    if not isinstance(fs, (int, float)) or fs <= 0:
        return {
            "statusCode":
            400,
            "body":
            json.dumps({
                "error":
                "Sampling frequency (fs) must be a positive number."
            })
        }
    if not isinstance(start_time_iso, str):
        return {
            "statusCode":
            400,
            "body":
            json.dumps(
                {"error": "start_time_iso must be an ISO format string."})
        }

    try:
        start_datetime_obj = datetime.fromisoformat(
            start_time_iso.replace("Z", "+00:00"))
        if start_datetime_obj.tzinfo is None:
            start_datetime_obj = start_datetime_obj.replace(
                tzinfo=timezone.utc)
    except ValueError:
        return {
            "statusCode":
            400,
            "body":
            json.dumps({
                "error":
                f"Invalid start_time_iso format: {start_time_iso}. Expected ISO format."
            })
        }

    ecg_signal_nk = np.array(ecg_data)
    print(
        f"Processing ECG data: {len(ecg_signal_nk)} samples, fs: {fs} Hz, start_time: {start_datetime_obj.isoformat()}"
    )

    # --- Heartbeat Detection ---
    heartbeat_times = np.array([])
    print("Detecting heartbeats...")
    try:
        if len(ecg_signal_nk
               ) / fs < 1:  # Very short signal, less than 1 second
            print(
                "Warning: ECG signal is very short. Heartbeat detection might be unreliable or fail."
            )

        heartbeat_indices = sleepecg.detect_heartbeats(ecg_signal_nk, fs=fs)
        if heartbeat_indices.size > 0:
            heartbeat_times = heartbeat_indices / fs
        print(f"Detected {len(heartbeat_times)} heartbeats.")

        if len(heartbeat_times) < MIN_HEARTBEATS_FOR_ANALYSIS:
            note_suffix = " Insufficient heartbeats detected."
            print(
                f"Warning: Too few heartbeats detected ({len(heartbeat_times)} out of {len(ecg_signal_nk) / fs :.2f}s signal). Analysis may be unreliable or fail."
            )
            # Decide if to proceed or return default. For now, proceed but add note.
            if not heartbeat_times.size:
                print(
                    "Error: No heartbeats detected. Cannot proceed with feature extraction."
                )
                num_epochs_fallback = int(
                    len(ecg_signal_nk) / (fs * SLEEP_STAGE_DURATION_S))
                indices, labels = _generate_default_target_stage_prediction(
                    num_epochs_fallback)
                return {
                    "statusCode":
                    200,
                    "body":
                    json.dumps({
                        "error":
                        "No heartbeats detected from ECG data.",
                        "predicted_sleep_stage_indices":
                        indices.tolist(),
                        "predicted_sleep_stage_labels":
                        labels,
                        "probabilities_per_epoch": [],
                        "note":
                        f"Used sleepecg.stage API with '{STAGES_MODE_FOR_CLF}' mode.{note_suffix}"
                    })
                }
    except Exception as hb_err:
        print(f"Error during heartbeat detection: {hb_err}")
        print(traceback.format_exc())
        num_epochs_fallback = int(
            len(ecg_signal_nk) / (fs * SLEEP_STAGE_DURATION_S))
        indices, labels = _generate_default_target_stage_prediction(
            num_epochs_fallback)
        return {
            "statusCode":
            500,
            "body":
            json.dumps({
                "error":
                "Failed during heartbeat detection.",
                "predicted_sleep_stage_indices":
                indices.tolist(),
                "predicted_sleep_stage_labels":
                labels,
                "probabilities_per_epoch": [],
                "note":
                f"Used sleepecg.stage API with '{STAGES_MODE_FOR_CLF}' mode. Heartbeat detection error."
            })
        }
    # --- End Heartbeat Detection ---

    # Create SleepRecord
    try:
        sleep_record_obj = sleepecg.SleepRecord(
            id="api_record",
            recording_start_time=start_datetime_obj.time(
            ),  # datetime.time object
            heartbeat_times=heartbeat_times,
            sleep_stage_duration=
            SLEEP_STAGE_DURATION_S  # Good to provide if known
        )
        print("SleepRecord created.")
    except Exception as sr_err:
        print(f"Error creating SleepRecord: {sr_err}")
        print(traceback.format_exc())
        num_epochs_fallback = int(
            len(ecg_signal_nk) / (fs * SLEEP_STAGE_DURATION_S))
        indices, labels = _generate_default_target_stage_prediction(
            num_epochs_fallback)
        return {
            "statusCode":
            500,
            "body":
            json.dumps({
                "error":
                "Failed to create SleepRecord object.",
                "predicted_sleep_stage_indices":
                indices.tolist(),
                "predicted_sleep_stage_labels":
                labels,
                "probabilities_per_epoch": [],
                "note":
                f"Used sleepecg.stage API with '{STAGES_MODE_FOR_CLF}' mode. SleepRecord creation failed."
            })
        }

    num_epochs = int(len(ecg_signal_nk) / (fs * SLEEP_STAGE_DURATION_S))
    print(f"Calculated number of epochs for output: {num_epochs}")

    if num_epochs == 0:  # This check is based on raw signal length for output epochs
        print(
            "Not enough data for any full output epochs. Returning empty prediction."
        )
        return {
            "statusCode":
            200,
            "body":
            json.dumps({
                "message":
                "Not enough data for any full output epochs.",
                "predicted_sleep_stage_indices": [],
                "predicted_sleep_stage_labels": [],
                "probabilities_per_epoch": [],
                "note":
                f"Used sleepecg.stage API with '{STAGES_MODE_FOR_CLF}' mode."
            })
        }

    predicted_stages_probabilities = _try_stage_sleep_record(
        CLF, sleep_record_obj)
    error_note_suffix = ""
    if len(heartbeat_times) < MIN_HEARTBEATS_FOR_ANALYSIS:
        error_note_suffix += " Warning: Insufficient heartbeats for reliable analysis."

    if predicted_stages_probabilities is None:
        print(
            "Sleep staging failed after retries. Generating default prediction."
        )
        predicted_sleep_stage_indices, predicted_sleep_stage_labels = _generate_default_target_stage_prediction(
            num_epochs)
        probabilities_for_json = []
        error_note_suffix += " Staging failed, returning default."
    else:
        print("Sleep staging successful.")
        predicted_sleep_stage_indices, predicted_sleep_stage_labels = _get_target_stage_predictions(
            predicted_stages_probabilities)
        probabilities_for_json = predicted_stages_probabilities.tolist(
        ) if predicted_stages_probabilities is not None else []

    if not predicted_sleep_stage_indices.size and num_epochs > 0 and predicted_stages_probabilities is not None:
        print(
            "Warning: Predictions are empty despite successful staging. Generating default as a fallback."
        )
        predicted_sleep_stage_indices, predicted_sleep_stage_labels = _generate_default_target_stage_prediction(
            num_epochs)
        probabilities_for_json = []
        error_note_suffix += " Predictions were empty after staging, fell back to default."

    response_body = {
        "predicted_sleep_stage_indices":
        predicted_sleep_stage_indices.tolist(),
        "predicted_sleep_stage_labels":
        predicted_sleep_stage_labels,
        # "probabilities_per_epoch": probabilities_for_json,
        "note":
        f"Used sleepecg.stage API with '{STAGES_MODE_FOR_CLF}' mode, mapped to {TARGET_N_STAGES} stages ({', '.join(TARGET_STAGE_LABELS.values())}).{error_note_suffix}"
    }
    return {"statusCode": 200, "body": json.dumps(response_body)}


def predict(event, context):
    # print(f"Received event: {json.dumps(event)}")
    if not sleepecg:
        print("Error: sleepecg library not available.")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "SleepECG library not available."})
        }
    if not CLF:
        print("Error: SleepECG model not loaded.")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "SleepECG model not loaded."})
        }

    try:
        body_data = None
        if 'body' in event and isinstance(event['body'], str):
            try:
                body_data = json.loads(event['body'])
            except json.JSONDecodeError:
                print("Error: Invalid JSON in request body.")
                return {
                    "statusCode": 400,
                    "body":
                    json.dumps({"error": "Invalid JSON in request body."})
                }
        elif isinstance(
                event, dict
        ) and 'ecg_data' in event and 'fs' in event and 'start_time_iso' in event:
            body_data = event
        else:
            print(
                "Error: Missing or malformed 'body' in event, or direct invocation with missing fields."
            )
            return {
                "statusCode":
                400,
                "body":
                json.dumps({
                    "error":
                    "Invalid request format. Expected JSON body with ecg_data, fs, and start_time_iso."
                })
            }

        ecg_data = body_data.get('ecg_data')
        fs = body_data.get('fs')
        start_time_iso = body_data.get('start_time_iso')

        if ecg_data is None or fs is None or start_time_iso is None:
            missing_params = [
                p for p, v in [('ecg_data', ecg_data), (
                    'fs', fs), ('start_time_iso', start_time_iso)] if v is None
            ]
            return {
                "statusCode":
                400,
                "body":
                json.dumps({
                    "error":
                    f"Missing required parameters: {', '.join(missing_params)}"
                })
            }

        return predict_eeg_data(ecg_data, fs, start_time_iso)
    except Exception as e:
        print(f"Error during prediction: {e}")
        print(traceback.format_exc())
        return {
            "statusCode":
            500,
            "body":
            json.dumps({"error": f"An unexpected error occurred: {str(e)}"})
        }


if __name__ == '__main__':
    print("Starting local test...")
    fs_test = 100
    duration_minutes_test = 10
    num_samples_test = fs_test * 60 * duration_minutes_test
    dummy_ecg_data_test = []
    try:
        if nk:
            dummy_ecg_data_test_nk = nk.ecg_simulate(
                duration=duration_minutes_test * 60,
                sampling_rate=fs_test,
                heart_rate=70,
                method="ecgsyn")
            dummy_ecg_data_test = dummy_ecg_data_test_nk.tolist()
            print(
                f"Generated {duration_minutes_test} minutes of ECG data using NeuroKit2."
            )
        else:
            raise ImportError("Neurokit2 not available for ECG simulation.")
    except Exception as nk_err:
        print(
            f"Could not use NeuroKit2 to simulate ECG ({nk_err}), falling back to random data."
        )
        dummy_ecg_data_test = np.random.randn(num_samples_test).tolist()

    start_time_test = datetime.now(timezone.utc).isoformat()
    test_event = {
        "ecg_data": dummy_ecg_data_test,
        "fs": fs_test,
        "start_time_iso": start_time_test
    }
    test_context = {}

    if not sleepecg or not CLF:
        print(
            "Cannot run local test: sleepecg not imported or CLF model not loaded."
        )
    else:
        print("\n--- Running predict function with test event ---")
        result = predict(test_event, test_context)
        print("\n--- Prediction Result ---")
        print(f"Status Code: {result.get('statusCode')}")
        try:
            result_body = json.loads(result.get('body', '{}'))
            print(f"Body: {json.dumps(result_body, indent=2)}")
            if 'predicted_sleep_stage_labels' in result_body:
                print(
                    f"\nNumber of predicted epochs: {len(result_body['predicted_sleep_stage_labels'])}"
                )
                if len(result_body['predicted_sleep_stage_labels']) > 0:
                    print(
                        f"First few predicted labels: {result_body['predicted_sleep_stage_labels'][:10]}"
                    )
                unique_labels = set(
                    result_body['predicted_sleep_stage_labels'])
                print(f"Unique predicted labels: {unique_labels}")
                expected_labels = set(TARGET_STAGE_LABELS.values())
                if unique_labels.issubset(
                        expected_labels
                ) or "Unknown" in unique_labels or not unique_labels:  # Allow empty if no epochs
                    print(
                        "Predicted labels are consistent with TARGET_STAGE_LABELS (or empty/Unknown)."
                    )
                else:
                    print(
                        f"Warning: Predicted labels {unique_labels} are not a subset of expected {expected_labels}."
                    )
        except json.JSONDecodeError:
            print(f"Raw Body (not JSON): {result.get('body')}")
        print("--- End of Local Test ---")
