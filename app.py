import chainlit as cl
from openai import OpenAI
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Inicializar el cliente de OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Diccionario para almacenar los datos de los usuarios
user_data = {}

def calcular_imc(peso, altura):
    """
    Calcula el Índice de Masa Corporal y retorna el valor y la categoría
    
    Args:
        peso: peso en kilogramos
        altura: altura en metros
    
    Returns:
        tuple: (valor IMC, categoría)
    """
    imc = peso / (altura * altura)
    
    if imc < 18.5:
        categoria = "Bajo peso"
    elif imc < 25:
        categoria = "Peso normal"
    elif imc < 30:
        categoria = "Sobrepeso"
    elif imc < 35:
        categoria = "Obesidad grado I"
    elif imc < 40:
        categoria = "Obesidad grado II"
    else:
        categoria = "Obesidad grado III"
    
    return round(imc, 2), categoria

@cl.on_chat_start
async def start():
    """Iniciar una nueva sesión de chat"""
    # Inicializar datos del usuario
    cl.user_session.set("estado", "inicio")
    
    # Mensaje de bienvenida
    await cl.Message(
        content="¡Hola! Soy tu asistente para calcular el Índice de Masa Corporal (IMC). "
                "El IMC es un indicador que relaciona el peso y la altura para identificar posibles problemas de peso. "
                "¿Te gustaría calcular tu IMC ahora?"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Manejar los mensajes del usuario"""
    # Obtener el estado actual de la conversación
    estado = cl.user_session.get("estado")
    
    # Obtener el mensaje del usuario
    mensaje_usuario = message.content
    
    # Usar OpenAI para procesar las respuestas del usuario
    if estado == "inicio":
        # Analizar si el usuario quiere calcular su IMC
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un asistente médico. Analiza si el usuario quiere calcular su IMC y responde con 'SI' o 'NO'."},
                {"role": "user", "content": mensaje_usuario}
            ],
            temperature=0.3,
        )
        
        decision = response.choices[0].message.content.strip().upper()
        
        if "SI" in decision:
            cl.user_session.set("estado", "solicitar_peso")
            await cl.Message(content="Para calcular tu IMC, necesito algunos datos. ¿Cuál es tu peso en kilogramos?").send()
        else:
            await cl.Message(content="Entiendo. Si en algún momento deseas calcular tu IMC, solo házmelo saber. El IMC es útil para evaluar si tu peso está en un rango saludable.").send()
            cl.user_session.set("estado", "inicio")
    
    elif estado == "solicitar_peso":
        # Procesar la entrada de peso
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un asistente médico. Extrae el valor numérico del peso en kilogramos del mensaje del usuario. Responde solo con el número."},
                {"role": "user", "content": mensaje_usuario}
            ],
            temperature=0.3,
        )
        
        try:
            peso = float(response.choices[0].message.content.strip().replace(',', '.'))
            cl.user_session.set("peso", peso)
            cl.user_session.set("estado", "solicitar_altura")
            await cl.Message(content=f"Gracias, he registrado tu peso como {peso} kg. Ahora, ¿cuál es tu altura en metros? (por ejemplo, 1.75)").send()
        except ValueError:
            await cl.Message(content="No pude entender el peso que me proporcionaste. Por favor, indica tu peso en kilogramos usando un número (por ejemplo: 70).").send()
    
    elif estado == "solicitar_altura":
        # Procesar la entrada de altura
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un asistente médico. Extrae el valor numérico de la altura en metros del mensaje del usuario. Responde solo con el número."},
                {"role": "user", "content": mensaje_usuario}
            ],
            temperature=0.3,
        )
        
        try:
            altura_text = response.choices[0].message.content.strip().replace(',', '.')
            altura = float(altura_text)
            
            # Verificar si la altura está en centímetros y convertir si es necesario
            if altura > 3:  # Probablemente está en centímetros
                altura = altura / 100
                
            peso = cl.user_session.get("peso")
            imc, categoria = calcular_imc(peso, altura)
            
            # Preparar recomendaciones con GPT
            prompt_recomendacion = f"""
            Eres un asistente médico especializado en nutrición. Un paciente tiene un IMC de {imc} que corresponde a la categoría: {categoria}.
            Proporciona:
            1. Una breve explicación de lo que significa su IMC (máximo 2 oraciones)
            2. Entre 2-3 recomendaciones generales de salud relacionadas con su categoría de IMC
            3. Recuerda que esto no reemplaza el consejo médico profesional
            Responde de manera concisa y amigable.
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt_recomendacion}],
                temperature=0.7,
            )
            
            recomendaciones = response.choices[0].message.content.strip()
            
            # Enviar resultados
            mensaje_resultado = f"""
            **Resultados de tu IMC**
            
            Peso: {peso} kg
            Altura: {altura} m
            IMC calculado: {imc}
            Categoría: {categoria}
            
            {recomendaciones}
            
            ¿Te gustaría calcular el IMC nuevamente con otros valores?
            """
            
            await cl.Message(content=mensaje_resultado).send()
            cl.user_session.set("estado", "resultado")
        except ValueError:
            await cl.Message(content="No pude entender la altura que me proporcionaste. Por favor, indica tu altura en metros usando un número decimal (por ejemplo: 1.75).").send()
    
    elif estado == "resultado":
        # Verificar si el usuario quiere calcular el IMC nuevamente
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un asistente médico. Analiza si el usuario quiere calcular su IMC nuevamente y responde con 'SI' o 'NO'."},
                {"role": "user", "content": mensaje_usuario}
            ],
            temperature=0.3,
        )
        
        decision = response.choices[0].message.content.strip().upper()
        
        if "SI" in decision:
            cl.user_session.set("estado", "solicitar_peso")
            await cl.Message(content="Perfecto, vamos a calcular el IMC nuevamente. ¿Cuál es tu peso en kilogramos?").send()
        else:
            await cl.Message(content="Entendido. Si necesitas calcular tu IMC en el futuro o tienes otras preguntas sobre salud, no dudes en preguntar. ¡Cuídate!").send()
            cl.user_session.set("estado", "inicio")