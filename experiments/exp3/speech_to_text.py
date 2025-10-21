from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage
from pydub import AudioSegment
import os
import io

def extract_audio_from_mp4(mp4_path, output_audio_path):
    """Trích xuất audio từ file MP4 và chuyển sang WAV"""
    print(f"Đang trích xuất audio từ {mp4_path}...")
    
    # Load video và extract audio
    audio = AudioSegment.from_file(mp4_path, format="mp4")
    
    # Convert sang mono và sample rate 16kHz (tối ưu cho Speech-to-Text)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    
    # Export sang WAV
    audio.export(output_audio_path, format="wav")
    print(f"Đã lưu audio vào {output_audio_path}")
    
    return output_audio_path

def upload_to_gcs(bucket_name, source_file_path, destination_blob_name):
    """Upload file lên Google Cloud Storage"""
    print(f"Đang upload {source_file_path} lên GCS...")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    blob.upload_from_filename(source_file_path)
    
    gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
    print(f"File đã được upload: {gcs_uri}")
    
    return gcs_uri

def transcribe_audio_gcs(gcs_uri, language_code="vi-VN"):
    """Chuyển đổi audio từ GCS sang text"""
    print(f"Đang chuyển đổi audio sang text (ngôn ngữ: {language_code})...")
    
    client = speech.SpeechClient()
    
    audio = speech.RecognitionAudio(uri=gcs_uri)
    
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
        enable_automatic_punctuation=True,  # Tự động thêm dấu câu
        enable_word_time_offsets=True,  # Lấy timestamp của từng từ
        model="default",  # Có thể dùng "video" cho video content
    )
    
    # Sử dụng long_running_recognize cho file dài
    operation = client.long_running_recognize(config=config, audio=audio)
    
    print("Đang xử lý... (có thể mất vài phút)")
    response = operation.result(timeout=600)
    
    return response

def transcribe_audio_local(audio_path, language_code="vi-VN"):
    """Chuyển đổi audio local sang text (cho file < 1 phút)"""
    print(f"Đang chuyển đổi audio sang text (ngôn ngữ: {language_code})...")
    
    client = speech.SpeechClient()
    
    with io.open(audio_path, "rb") as audio_file:
        content = audio_file.read()
    
    audio = speech.RecognitionAudio(content=content)
    
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
        enable_automatic_punctuation=True,
    )
    
    response = client.recognize(config=config, audio=audio)
    
    return response

def save_transcription(response, output_file):
    """Lưu kết quả transcription vào file"""
    with open(output_file, "w", encoding="utf-8") as f:
        for i, result in enumerate(response.results):
            alternative = result.alternatives[0]
            f.write(f"Transcript {i+1}:\n")
            f.write(f"{alternative.transcript}\n")
            f.write(f"Confidence: {alternative.confidence:.2%}\n\n")
            
            # Ghi timestamp nếu có
            if hasattr(alternative, 'words') and alternative.words:
                f.write("Word timestamps:\n")
                for word_info in alternative.words:
                    word = word_info.word
                    start_time = word_info.start_time.total_seconds()
                    end_time = word_info.end_time.total_seconds()
                    f.write(f"  {word}: {start_time:.2f}s - {end_time:.2f}s\n")
                f.write("\n")
    
    print(f"Kết quả đã được lưu vào {output_file}")

def main():
    """
    Main function - Có 2 cách sử dụng:
    
    Cách 1: Cho file ngắn (<1 phút) - Không cần GCS
    Cách 2: Cho file dài (>1 phút) - Cần upload lên GCS
    """
    
    # ===== CẤU HÌNH =====
    mp4_path = "Motoman dual arm robot compounding anti cancer drugs - YouTube.MP4"  # Đường dẫn file MP4 của bạn
    audio_path = "./exp3/extracted_audio.wav"
    output_file = "./exp3/transcription.txt"
    language_code = "ja-JP"  # Thay "en-US" cho tiếng Anh
    
    # Cho file dài - cần GCS
    use_gcs = True  # Đặt False nếu file < 1 phút
    bucket_name = "tuyennx_test"  # Tên GCS bucket của bạn
    blob_name = "audio/poc_proj_extracted_audio.wav"
    
    # ===== XỬ LÝ =====
    
    # Bước 1: Extract audio từ MP4
    extract_audio_from_mp4(mp4_path, audio_path)
    
    # Bước 2: Transcribe
    if use_gcs:
        # Upload lên GCS và transcribe
        gcs_uri = upload_to_gcs(bucket_name, audio_path, blob_name)
        response = transcribe_audio_gcs(gcs_uri, language_code)
    else:
        # Transcribe trực tiếp từ file local (chỉ cho file ngắn)
        response = transcribe_audio_local(audio_path, language_code)
    
    # Bước 3: Lưu kết quả
    save_transcription(response, output_file)
    
    # In transcript ra console
    print("\n===== KẾT QUẢ =====")
    for result in response.results:
        print(result.alternatives[0].transcript)

if __name__ == "__main__":
    # Thiết lập Google Cloud credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcs_credentials.json"
    
    main()
