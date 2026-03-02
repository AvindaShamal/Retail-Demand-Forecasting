import boto3


def download_from_s3(bucket_name, s3_key, local_path):
    s3_client = boto3.client("s3")
    try:
        s3_client.download_file(bucket_name, s3_key, local_path)
        print(f"File s3://{bucket_name}/{s3_key} downloaded to {local_path}")
    except Exception as e:
        print(f"Error downloading file from S3: {e}")
        
def upload_to_s3(file_path, bucket_name, s3_key):
    s3_client = boto3.client("s3")
    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
        print(f"File {file_path} uploaded to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
