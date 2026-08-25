"""
test_api.py
===========
Automated test script for Stage 8 FastAPI Backend.
Uses standard library urllib.request to send HTTP requests to a running FastAPI server.
"""

import json
import urllib.request
import urllib.error
import time

BASE_URL = "http://127.0.0.1:8000"

def test_root():
    print("\n[1] Testing GET / ...")
    req = urllib.request.Request(f"{BASE_URL}/")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print("    Response status:", resp.status)
        print("    Data:", data)
        assert resp.status == 200
        assert "message" in data

def test_health():
    print("\n[2] Testing GET /health ...")
    req = urllib.request.Request(f"{BASE_URL}/health")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print("    Response status:", resp.status)
        print("    Data:", data)
        assert resp.status == 200
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

def test_predict_single():
    print("\n[3] Testing POST /predict (Single Student)...")
    payload = {
        "study_hours": 8.0,
        "attendance_pct": 90.0,
        "prev_exam_score": 85.0,
        "assignments_done": 9,
        "sleep_hours": 7.5
    }
    json_data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        f"{BASE_URL}/predict",
        data=json_data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print("    Response status:", resp.status)
        print("    Predicted Score:", data["predicted_score"])
        print("    Grade:", data["grade"])
        print("    Interpretation:", data["interpretation"])
        assert resp.status == 200
        assert "predicted_score" in data
        assert data["grade"] == "A"

def test_validation_error():
    print("\n[4] Testing POST /predict validation error handling (study_hours > 24)...")
    payload = {
        "study_hours": 99.0,  # Invalid! Range is 0 to 24
        "attendance_pct": 90.0,
        "prev_exam_score": 85.0,
        "assignments_done": 9,
        "sleep_hours": 7.5
    }
    json_data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        f"{BASE_URL}/predict",
        data=json_data,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            print("    Unexpected success response!")
    except urllib.error.HTTPError as e:
        print("    Correctly received HTTP Error status:", e.code)
        err_body = json.loads(e.read().decode('utf-8'))
        print("    Validation Error Detail:", err_body["detail"][0]["msg"])
        assert e.code == 422  # Pydantic validation error code

def test_batch_predict():
    print("\n[5] Testing POST /predict/batch ...")
    payload = {
        "students": [
            {
                "study_hours": 9.0,
                "attendance_pct": 95.0,
                "prev_exam_score": 88.0,
                "assignments_done": 10,
                "sleep_hours": 7.5
            },
            {
                "study_hours": 2.0,
                "attendance_pct": 55.0,
                "prev_exam_score": 40.0,
                "assignments_done": 1,
                "sleep_hours": 5.0
            }
        ]
    }
    json_data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        f"{BASE_URL}/predict/batch",
        data=json_data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print("    Response status:", resp.status)
        print("    Total evaluated:", data["total_students"])
        for i, pred in enumerate(data["predictions"]):
            print(f"      Student {i+1}: Score={pred['predicted_score']}, Grade={pred['grade']}")
        assert resp.status == 200
        assert data["total_students"] == 2

if __name__ == "__main__":
    print("=" * 60)
    print("  FASTAPI BACKEND AUTOMATED INTEGRATION TESTS")
    print("=" * 60)
    
    # Wait for server to start if running script directly
    time.sleep(1)
    
    try:
        test_root()
        test_health()
        test_predict_single()
        test_validation_error()
        test_batch_predict()
        print("\n" + "=" * 60)
        print("  ✅ ALL API INTEGRATION TESTS PASSED PERFECTLY!")
        print("=" * 60)
    except Exception as err:
        print("\n❌ Test failed:", err)
