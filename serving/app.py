from flask import Flask, request, render_template, jsonify
import torch
import os
from inference_model import InferenceModel # `model.py`나 `inference_model.py`와 같은 파일에 정의된 클래스를 가져옵니다.
from dotenv import load_dotenv
import google.generativeai as genai

# 환경 변수에서 API 키를 불러옵니다.
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel()

app = Flask(__name__)

# 전역 상수
MODEL_PATH = "vgg_2class.pth"
CLASS_NAMES = ['cat', 'dog']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 애플리케이션 시작 시 모델 로드
try:
    # InferenceModel 클래스를 사용하여 모델 초기화 및 로딩을 한 번에 처리합니다.
    # 이 클래스는 포함(composition) 방식을 사용합니다.
    inference_model = InferenceModel(
        model_path=MODEL_PATH, 
        class_names=CLASS_NAMES, 
        device=DEVICE
    )
    # 모델 파일이 없는 경우, 모델 로드 실패를 명확히 알립니다.
    if not os.path.exists(MODEL_PATH):
        inference_model = None
    
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    inference_model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 파일 유효성 검사
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': '파일이 없거나 선택되지 않았습니다.'}), 400

    # 모델 로드 상태 확인
    if inference_model is None:
        return jsonify({'error': '모델이 로드되지 않았습니다.'}), 503

    try:
        file = request.files['file']
        image_bytes = file.read()
        
        # InferenceModel의 predict_image 메서드를 사용하여 예측을 수행합니다.
        prediction = inference_model.predict_image(image_bytes)
        
        if prediction is None:
            return jsonify({'error': '이미지 예측에 실패했습니다.'}), 500

        # 예측 결과를 바탕으로 Gemini API를 호출합니다.
        try:
            print(f"예측 결과: {prediction}, 타입: {type(prediction)}")
            prompt = (
                f"write a story about 750 characters long, about a {prediction}' "
                f"which is natural and creative."
            )
            gemini_response = gemini_model.generate_content(prompt).text
            return jsonify({"prediction": gemini_response})
        
        except Exception as e:
            print(f"Gemini API 호출 중 오류 발생: {e}")
            return jsonify({"error": f'Gemini API 오류: {e}'}), 500
        
    except Exception as e:
        print(f"예측 과정 중 오류 발생: {e}")
        return jsonify({'error': f'예측 오류: {e}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)