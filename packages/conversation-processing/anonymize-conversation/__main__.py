import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests
import urllib3
from boto3 import client as boto3_client
from botocore.exceptions import BotoCoreError, ClientError
from openai import OpenAI

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("log.log")],
)
logger = logging.getLogger(__name__)


class DigitalOceanBatchProcessor:
    def __init__(self):
        self.spaces_endpoint = os.getenv("DO_SPACES_ENDPOINT")
        self.spaces_key = os.getenv("DO_SPACES_ACCESS_KEY")
        self.spaces_secret = os.getenv("DO_SPACES_SECRET_KEY")
        self.spaces_bucket = os.getenv("DO_SPACES_BUCKET")
        self.agentai_endpoint = os.getenv("DO_AGENTAI_ENDPOINT")
        self.agentai_api_key = os.getenv("DO_AGENTAI_API_KEY")
        self.batch_size = int(os.getenv("PROCESSING_BATCH_SIZE", "100"))
        self.delete_processed = (
            os.getenv("DELETE_PROCESSED_FILES", "true").lower() == "true"
        )

        self.auth_url = os.getenv("ENDPOINT_AUTH_URL")
        self.user = os.getenv("ENDPOINT_AUTH_USER")
        self.password = os.getenv("ENDPOINT_AUTH_PASSWORD")
        self.key = os.getenv("ENDPOINT_AUTH_KEY")
        self.data_url = os.getenv("ENDPOINT_URL_TRANSCRIPTION")

        self._validate_config()

        self.s3_client = boto3_client(
            "s3",
            endpoint_url=self.spaces_endpoint,
            aws_access_key_id=self.spaces_key,
            aws_secret_access_key=self.spaces_secret,
            region_name="nyc3",
        )
        self.processed_files = []
        self.failed_files = []
        self._verify_connection()

        self.client = OpenAI(
            base_url=self.agentai_endpoint + "/api/v1/",
            api_key=self.agentai_api_key,
            timeout=300,
        )

    def _validate_config(self):
        """Validar configuraciones esenciales"""
        required_vars = [
            "DO_SPACES_ENDPOINT",
            "DO_SPACES_ACCESS_KEY",
            "DO_SPACES_SECRET_KEY",
            "DO_SPACES_BUCKET",
            "DO_AGENTAI_API_KEY",
            "ENDPOINT_AUTH_URL",
            "ENDPOINT_AUTH_USER",
            "ENDPOINT_AUTH_PASSWORD",
            "ENDPOINT_AUTH_KEY",
            "ENDPOINT_URL_TRANSCRIPTION",
        ]

        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required env vars:  {', '.join(missing_vars)}")

    def _verify_connection(self):
        """Verificar que la conexión a Spaces funciona"""
        try:
            self.s3_client.head_bucket(Bucket=self.spaces_bucket)
            logger.info(f"Successfully connected to bucket {self.spaces_bucket}")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                raise Exception(f"Bucket {self.spaces_bucket} does not exist")
            elif error_code == "403":
                raise Exception(f"Access denied to bucket {self.spaces_bucket}")
            else:
                raise Exception(f"Error connecting to Spaces: {str(e)}")

    def list_json_files(self, prefix: str = "") -> List[str]:
        logger.info(f"Listando archivos JSON en el bucket {prefix}")
        """Listar archivos JSON en el bucket con límite de batch_size"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.spaces_bucket, Prefix=prefix, MaxKeys=self.batch_size
            )

            return [
                obj["Key"]
                for obj in response.get("Contents", [])
                if obj["Key"].lower().endswith(".json")
                and "/failed/" not in obj["Key"].lower()
            ]
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Error listing files: {str(e)}")
            return []

    def get_json_from_spaces(self, object_key: str) -> Optional[Dict[str, Any]]:
        """Descargar JSON desde Spaces"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.spaces_bucket, Key=object_key
            )
            return json.loads(response["Body"].read().decode("utf-8"))
        except (ClientError, BotoCoreError, json.JSONDecodeError) as e:
            logger.error(f"Error downloading {object_key}: {str(e)}")
            return None

    def process_conversation_data(
        self, conversation_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Procesa los datos de una conversación para extraer el texto completo y los segmentos."""
        try:
            conversation_id = conversation_data.get("conversation_id")
            started_date = conversation_data.get("started_date")
            duration_seconds = conversation_data.get("conversation_total_time")
            call_phone_number = conversation_data.get("to")
            client_phone_number = conversation_data.get("origen")
            agent_info = conversation_data.get("agent", [{}])[0]
            agent_name = f"{agent_info.get('first_name', '')} {agent_info.get('last_name', '')}".strip()
            agent_email = agent_info.get("email", "")
            channel = conversation_data.get("channel")
            skill = conversation_data.get("skill", "")

            full_text = ""
            segments = []
            segment_id = 1

            for message in conversation_data.get("messages", []):
                message_type = message.get("message_type")
                text = message.get("text")
                message_text = text.strip() if text is not None else ""
                timestamp = message.get("timestamp", None)

                if message_type in ["custom", "reply"]:
                    continue

                if not message_text:
                    continue

                message_direction = message.get("message_direction")

                if message_direction == "outbound":
                    speaker = "Agent"
                elif message_direction == "inbound":
                    speaker = "Customer"
                else:
                    speaker = ""
                    logger.warning(f"message direction: {message}")
                    continue

                anonymized_text = self.anonymize_transcription(message_text, agent_name)
                if not anonymized_text:
                    message_id = message.get("message_id")
                    logger.error(f"Error anonymizing the message{message_id}.")
                    anonymized_text = message_text

                full_text += f"{anonymized_text} "

                segments.append(
                    {
                        "SegmentId": segment_id,
                        "Date": timestamp,
                        "Speaker": speaker,
                        "Text": anonymized_text,
                        "StartTime": 0,
                        "EndTime": 0,
                    }
                )
                segment_id += 1

            processed_data = {
                "Conversationid": conversation_id,
                "FileName": f"{conversation_id}.json",
                "ConversationDate": started_date,
                "DurationSeconds": duration_seconds,
                "CallPhoneNumber": call_phone_number,
                "ContactPhoneNumber": client_phone_number,
                "Channel": channel,
                "Skill": skill,
                "Agent": {"Name": agent_name, "Email": agent_email},
                "Transcription": {"FullText": full_text, "Segments": segments},
            }

            return processed_data

        except Exception as e:
            logger.error(f"Error processing conversation data: {str(e)}", exc_info=True)
            logger.error(f"data {processed_data}".encode("ascii", "replace").decode())
            return None

    def anonymize_transcription(
        self, full_text: str, agent_name: str
    ) -> Optional[dict]:
        """Anonimiza una transcripcion reemplazando datos sensibles con 'XXXXX'.
        - `full_text`: Conversacion completa a anonimizar."""

        try:
            time.sleep(1)
            undesired_comments = [
                "No hay texto que anonimizar.",
                "No hay texto que deba ser anonimizado.",
                "No hay texto que procesar.",
                "No anonimizar",
                "Texto exactamente igual",
                "Texto procesado con anonimización",
            ]

            response = self.client.chat.completions.create(
                model="llama-3.3-instruct-70b",
                messages=[{"role": "user", "content": full_text}],
                max_tokens=4096,
                extra_body={"include_retrieval_info": False},
            )

            anonymized_response = response.choices[0].message.content.strip()

            cleaned_original_input = full_text.strip()
            for undesired_comment in undesired_comments:
                if undesired_comment in anonymized_response:
                    return cleaned_original_input

            return anonymized_response

        except Exception as e:
            logger.error(f"Error when calling DigitalOcean GenAI: {str(e)}")
            return None

    def get_token(self):
        """Obtener el Bearer Token"""
        try:
            response = requests.post(
                self.auth_url,
                json={
                    "user": self.user,
                    "password": self.password,
                    "key": self.key,
                    "access": "IbangMiddlewareApi",
                },
                verify=False,
            )

            if response.status_code == 200:
                return response.json().get("auth_token")
            else:
                logger.error(
                    f"Error getting token: {response.status_code} - {response.text}"
                )
                return False
        except requests.RequestException as e:
            logger.error(f"Connection error when obtaining token: {str(e)}")
            return False

    def send_to_endpoint(self, processed_data: Dict[str, Any]) -> bool:
        """
        Enviar datos procesados al endpoint de Holamigo con autenticación.
        - `processed_data`: Estructura generada por `process_conversation_data`.
        """
        token = self.get_token()
        if not token:
            logger.error("The token could not be obtained. Canceling sending.")
            return False

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.data_url, json=processed_data, headers=headers, verify=False
            )
            response_data = response.json()

            if response.status_code == 200 and response_data.get("status"):
                logger.info(
                    f"Datos enviados correctamente para {processed_data['FileName']}"
                )
                return True
            else:
                messageError = response_data.get("message")
                logger.error(
                    f"Error al enviar datos: {response.status_code} - {response.text} - {messageError}"
                )
                return False
        except requests.RequestException as e:
            logger.error(f"Error de conexión al enviar datos: {str(e)}")
            return False

    def delete_file(self, object_key: str) -> bool:
        """Eliminar archivo procesado"""
        if not self.delete_processed:
            return True

        try:
            self.s3_client.delete_object(Bucket=self.spaces_bucket, Key=object_key)
            return True
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Error deleting {object_key}: {str(e)}")
            return False

    def move_to_failed(self, object_key: str):
        """Mover archivo a la carpeta Failed dentro de la estructura de fecha"""
        try:
            # Mantener el prefijo de la fecha y agregar la carpeta Failed
            base_folder = os.path.dirname(object_key)
            failed_key = f"{base_folder}/failed/{os.path.basename(object_key)}"

            self.s3_client.copy_object(
                Bucket=self.spaces_bucket,
                CopySource={"Bucket": self.spaces_bucket, "Key": object_key},
                Key=failed_key,
            )
            self.s3_client.delete_object(Bucket=self.spaces_bucket, Key=object_key)
            logger.info(f"Moved {object_key} to {failed_key}")
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Error moving {object_key} to Failed: {str(e)}")

    def process_batch(self):
        """Procesar un lote de archivos en paralelo"""
        dateCurrent = datetime.now()
        datePrevious = dateCurrent - timedelta(days=40)
        dateCurrentFormatted = dateCurrent.strftime("%Y-%m-%d")
        datePreviousFormatted = datePrevious.strftime("%Y-%m-%d")

        prefixes = [
            f"conversations/{datePreviousFormatted}",
            f"conversations/{dateCurrentFormatted}",
        ]

        files = []
        for prefix in prefixes:
            files.extend(self.list_json_files(prefix))

        if not files:
            logger.error("No files found to process")
            return

        logger.info(f"Processing batch of {len(files)} files")

        # Usar ThreadPoolExecutor para procesar en paralelo
        with ThreadPoolExecutor(
            max_workers=3
        ) as executor:  # Ajusta `max_workers` según tus necesidades
            futures = {
                executor.submit(self.process_and_send_file, file_key): file_key
                for file_key in files
            }

            for future in as_completed(futures):
                file_key = futures[future]
                try:
                    result = future.result()
                    if result:
                        self.processed_files.append(file_key)
                    else:
                        self.failed_files.append(file_key)
                except Exception as e:
                    logger.error(f"Unexpected error processing {file_key}: {str(e)}")
                    self.failed_files.append(file_key)

        logger.info(
            f"Batch processing completed. Processed: {len(self.processed_files)}, Failed: {len(self.failed_files)}"
        )

    def process_and_send_file(self, file_key: str) -> bool:
        """Procesar y enviar un archivo individual"""
        try:
            logger.info(f"Procesando archivo: {file_key}")
            conversation_data = self.get_json_from_spaces(file_key)
            if not conversation_data:
                logger.error(f"Failed to download {file_key}")
                return False

            processed_data = self.process_conversation_data(conversation_data)
            if not processed_data:
                logger.error(f"Error procesando los datos del archivo {file_key}")
                return False

            if self.send_to_endpoint(processed_data):
                if self.delete_file(file_key):
                    logger.info(f"Successfully processed and deleted {file_key}")
                    return True
            else:
                logger.error(f"Failed to send {file_key} to endpoint")
                self.move_to_failed(file_key)
                return False
        except Exception as e:
            logger.error(f"Unexpected error processing {file_key}: {str(e)}")
            self.move_to_failed(file_key)
            return False


def main(event=None, context=None):
    """Función principal para ejecución programada"""
    try:
        logger.info("Starting batch processing")
        processor = DigitalOceanBatchProcessor()
        processor.process_batch()

        return {
            "status": "success",
            "processed": processor.processed_files,
            "failed": processor.failed_files,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# Comentar cuando se termine el desarrollo
if __name__ == "__main__":
    result = main()
    print(result)
