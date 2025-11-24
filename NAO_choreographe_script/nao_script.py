#ESTE SCRIPT SOLO CORRE EN CHOREOGRAPHE PARA EL ROBOT NAO
#Funciona en una versión de python antigua

# -*- coding: utf-8 -*-
import sys
import time
import json
import urllib2
from naoqi import ALProxy

class MyClass(GeneratedClass):
    def __init__(self):
        GeneratedClass.__init__(self)
        self.video_service = ALProxy("ALVideoDevice")
        self.tts = ALProxy("ALTextToSpeech")

        # AJUSTA TU IP AQUI
        self.api_url = "http://10.32.50.128:8000/predict_lite/"

    def onLoad(self):
        pass

    def onUnload(self):
        pass

    def onInput_onStart(self):
        self.tts.say("Observando entorno.")

        cameraID = 1     # 0=Top, 1=Bottom
        resolution = 2   # VGA
        colorSpace = 11  # RGB
        fps = 30

        subscriberID = self.video_service.subscribeCamera("lite_client", cameraID, resolution, colorSpace, fps)

        try:
            # Capturar imagen
            naoImage = self.video_service.getImageRemote(subscriberID)
            self.tts.say("Procesando imagen.")

            if naoImage is None:
                self.log("Error: No se pudo obtener imagen.")
                return

            imageWidth = naoImage[0]
            imageHeight = naoImage[1]
            pixelData = naoImage[6]

            self.log("Imagen capturada: " + str(imageWidth) + "x" + str(imageHeight))

            # Preparar Peticion
            req = urllib2.Request(self.api_url, data=pixelData)
            req.add_header('Content-Type', 'application/octet-stream')
            req.add_header('X-Image-Width', str(imageWidth))
            req.add_header('X-Image-Height', str(imageHeight))

            self.log("Conectando al servidor...")

            # Enviar y recibir respuesta
            response = urllib2.urlopen(req, timeout=10)
            response_text = response.read()

            # Procesar JSON
            data = json.loads(response_text)

            if data.get("success") == True:
                # Obtenemos el label y lo convertimos a string UTF-8 para evitar errores de acentos
                raw_label = data.get("label")
                if isinstance(raw_label, unicode):
                    label = raw_label.encode('utf-8')
                else:
                    label = str(raw_label)

                conf = int(data.get("confidence", 0) * 100)

                # --- LÓGICA DE CONTENEDORES ---
                # Definimos mensaje por defecto
                mensaje_objeto = "He detectado " + label
                mensaje_contenedor = "No estoy seguro de dónde se recicla esto."

                # Convertimos a minúsculas para comparar más fácil
                lbl_lower = label.lower()

                if "plastico" in lbl_lower or "plástico" in lbl_lower:
                    mensaje_contenedor = "Esto va en el contenedor amarillo."

                elif "metal" in lbl_lower:
                    mensaje_contenedor = "El metal va en el contenedor amarillo."

                elif "vidrio" in lbl_lower:
                    mensaje_contenedor = "El vidrio se tira en el contenedor verde."

                elif "papel" in lbl_lower or "carton" in lbl_lower:
                    mensaje_contenedor = "El papel va en el contenedor azul."

                elif "biodegradable" in lbl_lower:
                    mensaje_contenedor = "Esto es orgánico, va al contenedor marrón."

                elif "no reciclable" in lbl_lower:
                    mensaje_contenedor = "Esto no es reciclable, va al contenedor gris."

                # --- HABLA DEL ROBOT ---
                self.log("Resultado: " + label)
                self.tts.say(mensaje_objeto)

                # Pequeña pausa para naturalidad
                time.sleep(0.5)

                self.tts.say(mensaje_contenedor)

                self.onStopped(str(label))

            else:
                error_msg = data.get("error", "Desconocido")
                self.log("Error API: " + str(error_msg))
                self.tts.say("Hubo un error en el reconocimiento.")
                self.onStopped("Error")

        except Exception as e:
            self.log("Error critico: " + str(e))
            self.tts.say("Ocurrió un error de conexión.")
            self.onStopped("Error")
        finally:
            self.video_service.unsubscribe(subscriberID)

    def onInput_onStop(self):
        self.onUnload()
        self.onStopped()