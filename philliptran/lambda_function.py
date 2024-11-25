import base64
import boto3
import json
import os
import random

# Initialize AWS clients
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
s3_client = boto3.client("s3")

def lambda_handler(event, context):
    # Get the S3 bucket name and model ID from environment variables
    bucket_name = os.environ.get("S3_BUCKET_NAME")
    model_id = os.environ.get("MODEL_ID")
    
    # Parse the request body to get the prompt
    try:
        body = json.loads(event["body"])
        prompt = body["prompt"]
    except (KeyError, json.JSONDecodeError):
        return {
            "statusCode": 400,
            "body": json.dumps({"message": "Invalid request. 'prompt' is required."})
        }
    
    # Generate a random seed and S3 path for the image
    seed = random.randint(0, 2147483647)
    s3_image_path = f"generated_images/titan_{seed}.png"
    
    # Prepare the request payload for the Bedrock model
    native_request = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {"text": prompt},
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "standard",
            "cfgScale": 8.0,
            "height": 1024,
            "width": 1024,
            "seed": seed,
        }
    }
    
    try:
        # Invoke the Bedrock model
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(native_request)
        )
        model_response = json.loads(response["body"].read())
        
        # Extract and decode the Base64 image data
        base64_image_data = model_response["images"][0]
        image_data = base64.b64decode(base64_image_data)
        
        # Upload the image to S3
        s3_client.put_object(Bucket=bucket_name, Key=s3_image_path, Body=image_data)
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"message": "Failed to generate or upload the image.", "error": str(e)})
        }
    
    # Return a successful response
    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Image generated and stored successfully.",
            "s3_url": f"s3://{bucket_name}/{s3_image_path}"
        })
    }
