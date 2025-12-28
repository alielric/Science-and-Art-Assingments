{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "authorship_tag": "ABX9TyO5qunCANx5xWVUbquVc7lM",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/alielric/Science-and-Art-Assingments/blob/main/final/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import streamlit as st\n",
        "from transformers import pipeline\n",
        "import torch\n",
        "from diffusers import StableDiffusionPipeline\n",
        "from PIL import Image, ImageOps, ImageFilter\n",
        "import io\n",
        "\n",
        "st.set_page_config(page_title=\"GenAI Art & Chat Studio\", layout=\"wide\")\n",
        "\n",
        "st.title(\"🎨 GenAI Art & Chat Studio (Week 12 Final)\")\n",
        "\n",
        "@st.cache_resource\n",
        "def load_chat_model():\n",
        "    return pipeline(\"text2text-generation\", model=\"google/flan-t5-base\")\n",
        "\n",
        "@st.cache_resource\n",
        "def load_art_model():\n",
        "    model_id = \"runwayml/stable-diffusion-v1-5\"\n",
        "    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)\n",
        "    if torch.cuda.is_available():\n",
        "        pipe = pipe.to(\"cuda\")\n",
        "    return pipe\n",
        "\n",
        "mode = st.sidebar.radio(\"Mod Seçin:\", [\"Chat Mode\", \"Art Mode\"])\n",
        "\n",
        "if mode == \"Chat Mode\":\n",
        "    st.subheader(\"💬 AI Chatbot\")\n",
        "    chat_model = load_chat_model()\n",
        "    user_input = st.text_input(\"Bir şey sorun:\")\n",
        "    if st.button(\"Gönder\") and user_input:\n",
        "        response = chat_model(user_input, max_length=100)\n",
        "        st.write(\"**AI:**\", response[0]['generated_text'])\n",
        "\n",
        "else:\n",
        "    st.subheader(\"🖼️ Generative Art & Filters\")\n",
        "    art_model = load_art_model()\n",
        "    prompt = st.text_input(\"Görsel açıklaması:\", \"A futuristic city in watercolor style\")\n",
        "\n",
        "    if st.button(\"Görsel Oluştur\"):\n",
        "        image = art_model(prompt).images[0]\n",
        "        st.session_state['current_img'] = image\n",
        "\n",
        "    if 'current_img' in st.session_state:\n",
        "        filter_type = st.selectbox(\"Filtre Seçin:\", [\"Orijinal\", \"Siyah Beyaz\", \"Blur\", \"Kenar Belirleme\"])\n",
        "        img = st.session_state['current_img']\n",
        "        if filter_type == \"Siyah Beyaz\": img = ImageOps.grayscale(img)\n",
        "        elif filter_type == \"Blur\": img = img.filter(ImageFilter.BLUR)\n",
        "        elif filter_type == \"Kenar Belirleme\": img = img.filter(ImageFilter.FIND_EDGES)\n",
        "        st.image(img)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "AJyqyRSf82LA",
        "outputId": "f39e6181-7f8d-406b-c95f-4f3b008e0226"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Flax classes are deprecated and will be removed in Diffusers v1.0.0. We recommend migrating to PyTorch classes or pinning your version of Diffusers.\n",
            "Flax classes are deprecated and will be removed in Diffusers v1.0.0. We recommend migrating to PyTorch classes or pinning your version of Diffusers.\n",
            "2025-12-28 17:13:46.405 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:46.406 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:46.460 \n",
            "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
            "  command:\n",
            "\n",
            "    streamlit run /usr/local/lib/python3.12/dist-packages/colab_kernel_launcher.py [ARGUMENTS]\n",
            "2025-12-28 17:13:46.460 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:46.462 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:46.465 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:46.465 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:46.467 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:46.468 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:46.469 Session state does not function when running a script without `streamlit run`\n",
            "2025-12-28 17:13:46.471 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:46.473 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:46.474 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:46.475 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:46.476 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:46.477 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:46.478 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:46.479 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:46.480 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:46.481 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:46.983 Thread 'Thread-3': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:46.984 Thread 'Thread-3': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:46.985 Thread 'Thread-3': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: \n",
            "The secret `HF_TOKEN` does not exist in your Colab secrets.\n",
            "To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.\n",
            "You will be able to reuse this secret in all of your notebooks.\n",
            "Please note that authentication is recommended but still optional to access public models or datasets.\n",
            "  warnings.warn(\n",
            "Device set to use cuda:0\n",
            "2025-12-28 17:13:49.283 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:49.283 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:49.285 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:49.286 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:49.288 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:49.289 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:49.291 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:49.293 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:49.294 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:49.295 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:49.297 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:49.298 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:49.300 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:49.301 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:49.302 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:13:49.303 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ]
    }
  ]
}