# test_multimodal.py
from vcab import Model, save_video_stream_predictions_v3
import os

def test_multimodal():
    try:
        # Input video path
        video_path = "C:/Users/aarya/autism/VCAB/videos/normal.mp4"

        # Output video path
        output_path = "C:/Users/aarya/autism/VCAB/output/processed_v_normal.mp4"

        # Make sure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Run multimodal prediction
        actions, emotions, autism, autism_percentage, video_output = Model().predict_stream_emotion(
            video_path=video_path
        )

        # Save video with actions and autism predictions using v3
        save_video_stream_predictions_v3(
            video_path=video_path,
            action_predictions=actions,
            autism_predictions=autism,
            output_path=output_path
        )

        print(f"\n✅ Autism likelihood: {autism_percentage:.2f}%")
        print(f"🎥 Processed video saved at: {output_path}")

    except Exception as e:
        print(f"❌ Exception occurred: {e}")


# Run script directly
if __name__ == "__main__":
    test_multimodal()
