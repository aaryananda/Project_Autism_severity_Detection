# test_multimodal.py (debug / robust version)
import os
import traceback
from vcab import Model, save_video_stream_predictions_v3

THRESHOLD = 0.5  # probability threshold to mark "symptoms"

def pretty_print_autism_info(autism_predictions):
    """
    Accepts many possible forms of autism_predictions:
      - dict(time -> "symptoms"/"no_symptoms")
      - dict(time -> numeric score)
      - list of tuples, etc.
    Prints a summary and returns counts.
    """
    # Normalize to list of (time, value) pairs
    items = []
    if autism_predictions is None:
        print("autism_predictions is None")
        return 0, 0, items

    if isinstance(autism_predictions, dict):
        items = list(autism_predictions.items())
    elif isinstance(autism_predictions, list):
        # maybe list of (time, val) pairs
        try:
            items = [(float(t), v) for t, v in autism_predictions]
        except Exception:
            items = [(i, v) for i, v in enumerate(autism_predictions)]
    else:
        # unknown structure: try to iterate
        try:
            items = list(enumerate(autism_predictions))
        except Exception:
            print("Unknown autism_predictions type:", type(autism_predictions))
            return 0, 0, []

    total = len(items)
    symptom_count = 0
    no_symptom_count = 0
    numeric_scores = []

    # inspect some sample items
    print("Sample autism_predictions items (up to 10):")
    for s in items[:10]:
        print(" ", s)

    for time, val in items:
        # if value is string markers
        if isinstance(val, str):
            v_lower = val.lower()
            if "symptom" in v_lower or "autism" in v_lower or "yes" in v_lower or "1" == v_lower:
                symptom_count += 1
            else:
                no_symptom_count += 1
        else:
            # try numeric
            try:
                score = float(val)
                numeric_scores.append((time, score))
                if score >= THRESHOLD:
                    symptom_count += 1
                else:
                    no_symptom_count += 1
            except Exception:
                # fallback: treat as symptom if truthy
                if val:
                    symptom_count += 1
                else:
                    no_symptom_count += 1

    # If numeric scores are present, compute average
    avg_score = None
    if numeric_scores:
        avg_score = sum(s for _, s in numeric_scores) / len(numeric_scores)

    print("\nSummary:")
    print(f" - Total samples: {total}")
    print(f" - Symptom frames (by threshold/rule): {symptom_count}")
    #print(f" - No-symptom frames: {no_symptom_count}")
    if avg_score is not None:
        print(f" - Average numeric score (where available): {avg_score:.4f}")

    return symptom_count, no_symptom_count, items

def test_multimodal():
    try:
        video_path = "C:/Users/aarya/autism/VCAB/videos/Arm_Flap_0.mp4"
        output_path = "C:/Users/aarya/autism/VCAB/output/Arm_Flap_em.mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print("Instantiating model...")
        model = Model()
        print("Calling predict_stream_emotion(...)")
        # call the pipeline; keep this wrapped so we can inspect exceptions
        actions, emotions, autism_predictions, autism_percentage, video_output = model.predict_stream_emotion(
            video_path=video_path
        )

        print("\n--- RETURNED TYPES & LENGTHS ---")
        print("actions type:", type(actions), "len (if applicable):", getattr(actions, "__len__", lambda: "N/A")())
        print("emotions type:", type(emotions), "len (if applicable):", getattr(emotions, "__len__", lambda: "N/A")())
        print("autism_predictions type:", type(autism_predictions))
        print("autism_percentage type:", type(autism_percentage), "value:", autism_percentage)
        print("video_output:", video_output)

        # Save processed video using returned video_output if available,
        # otherwise fallback to output_path
        save_target = video_output if video_output else output_path
        print(f"\nSaving annotated video to: {save_target}")
        try:
            save_video_stream_predictions_v3(
                video_path=video_path,
                action_predictions=actions,
                autism_predictions=autism_predictions,
                output_path=save_target
            )
        except Exception as e:
            print("save_video_stream_predictions_v3 raised an exception:")
            traceback.print_exc()

        # Print autism percentage
        try:
            print(f"\n✅ Autism likelihood: {float(autism_percentage):.2f}%")
        except Exception:
            print("autism_percentage could not be converted to float (value: {})".format(autism_percentage))

        # If autism_percentage == 0 (or very small), say no autism
        try:
            if float(autism_percentage) == 0:
                print("\n🧠 No Autism Detected (autism_percentage == 0).")
            else:
                print("\n⚠️ Autism Detected (autism_percentage > 0).")
                if(float(autism_percentage)<50):
                    print("\nSeverity Level:- Mild .")
                elif(float(autism_percentage)>50 & float(autism_percentage)<100):
                    print("\nSeverity Level:- Moderate .")
                else:
                    
                    print("\nSeverity Level:- Severe.")



                    
        except Exception:
            print("\nCould not compare autism_percentage to zero.")

        # Detailed autism info
        symptom_count, no_symptom_count, items = pretty_print_autism_info(autism_predictions)

        # Print a timeline of detected symptom times (if timestamps present)
        print("\nDetailed timeline of detected symptom times (if available):")
        showed = 0
        for time, val in items:
            is_symptom = False
            if isinstance(val, str):
                if "symptom" in val.lower() or "autism" in val.lower():
                    is_symptom = True
            else:
                try:
                    if float(val) >= THRESHOLD:
                        is_symptom = True
                except Exception:
                    if val:
                        is_symptom = True

            if is_symptom:
                try:
                    print(f" - Time {float(time):.2f}s -> {val}")
                except Exception:
                    print(f" - Time {time} -> {val}")
                showed += 1
            if showed >= 200:
                print(" ... (stopping listing after 200 entries)")
                break

        print("\nDone.")

    except Exception as e:
        print("Top-level exception occurred:")
        traceback.print_exc()
        print("\nIf this still fails, please copy & paste the full traceback shown above here so I can debug further.")

if __name__ == "__main__":
    test_multimodal()
