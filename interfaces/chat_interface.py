import gradio as gr

from services.conversation_service import (
    new_conversation_state,
    process_message
)


WELCOME_MESSAGE = (
    '¡Hola! Soy tu asistente de evaluación crediticia. '
)


def send_message(
    message: str,
    history: list[dict] | None,
    state: dict | None
) -> tuple[str, list[dict], dict]:
    current_history = list(history or [])

    if not message.strip():
        return '', current_history, state or new_conversation_state()

    current_history.append({
        'role': 'user',
        'content': message
    })

    response, updated_state = process_message(
        message,
        state
    )

    current_history.append({
        'role': 'assistant',
        'content': response
    })

    return '', current_history, updated_state


def reset_conversation() -> tuple[list[dict], dict]:
    return (
        [{
            'role': 'assistant',
            'content': WELCOME_MESSAGE
        }],
        new_conversation_state()
    )


with gr.Blocks(title='Evaluación de Riesgo Crediticio') as demo:
    gr.Markdown(
        '''
# Evaluación de Riesgo Crediticio
Conversa con el asistente para analizar tu solicitud.
        '''.strip()
    )

    conversation_state = gr.State(new_conversation_state())

    chatbot = gr.Chatbot(
        value=[{
            'role': 'assistant',
            'content': WELCOME_MESSAGE
        }],
        height=520,
        layout='bubble',
        buttons=['copy']
    )

    message_box = gr.Textbox(
        placeholder='Escribe aquí...',
        show_label=False,
        lines=2
    )

    with gr.Row():
        send_button = gr.Button(
            'Enviar',
            variant='primary'
        )
        reset_button = gr.Button('Nueva evaluación')

    send_button.click(
        fn=send_message,
        inputs=[
            message_box,
            chatbot,
            conversation_state
        ],
        outputs=[
            message_box,
            chatbot,
            conversation_state
        ]
    )

    message_box.submit(
        fn=send_message,
        inputs=[
            message_box,
            chatbot,
            conversation_state
        ],
        outputs=[
            message_box,
            chatbot,
            conversation_state
        ]
    )

    reset_button.click(
        fn=reset_conversation,
        outputs=[
            chatbot,
            conversation_state
        ]
    )


if __name__ == '__main__':
    demo.launch(
        server_name='127.0.0.1',
        server_port=7860
    )
