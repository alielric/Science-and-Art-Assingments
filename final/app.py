{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "authorship_tag": "ABX9TyO9cTFTdjlsNk/u97J/0Hb+",
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
        "# Sayfa yapılandırması\n",
        "st.set_page_config(page_title=\"GenAI Art & Chat Studio\", layout=\"wide\")\n",
        "st.title(\"🎨 GenAI Art & Chat Studio (Week 12 Final)\")\n",
        "\n",
        "# Model yükleme fonksiyonları\n",
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
        "# Mod seçimi\n",
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
        "        with st.spinner(\"Görsel oluşturuluyor...\"):\n",
        "            image = art_model(prompt).images[0]\n",
        "            st.session_state['current_img'] = image\n",
        "\n",
        "    if 'current_img' in st.session_state:\n",
        "        filter_type = st.selectbox(\"Filtre Uygula (Week 11):\", [\"Orijinal\", \"Siyah Beyaz\", \"Bulanık\", \"Kenar Belirleme\"])\n",
        "        img = st.session_state['current_img']\n",
        "        if filter_type == \"Siyah Beyaz\": img = ImageOps.grayscale(img)\n",
        "        elif filter_type == \"Bulanık\": img = img.filter(ImageFilter.BLUR)\n",
        "        elif filter_type == \"Kenar Belirleme\": img = img.filter(ImageFilter.FIND_EDGES)\n",
        "        st.image(img, caption=f\"Mod: {filter_type}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zRaINssJ9ZzL",
        "outputId": "bdc95d01-b9dc-409f-d99f-f4a0aca57a6b"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Flax classes are deprecated and will be removed in Diffusers v1.0.0. We recommend migrating to PyTorch classes or pinning your version of Diffusers.\n",
            "Flax classes are deprecated and will be removed in Diffusers v1.0.0. We recommend migrating to PyTorch classes or pinning your version of Diffusers.\n",
            "2025-12-28 17:16:39.289 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:39.294 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:39.376 \n",
            "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
            "  command:\n",
            "\n",
            "    streamlit run /usr/local/lib/python3.12/dist-packages/colab_kernel_launcher.py [ARGUMENTS]\n",
            "2025-12-28 17:16:39.377 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:39.379 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:39.387 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:39.391 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:39.393 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:39.397 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:39.399 Session state does not function when running a script without `streamlit run`\n",
            "2025-12-28 17:16:39.401 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:39.404 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:39.405 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:39.409 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:39.413 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:39.413 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:39.415 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:39.416 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:39.418 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:39.420 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:39.924 Thread 'Thread-3': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:39.926 Thread 'Thread-3': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:39.930 Thread 'Thread-3': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: \n",
            "The secret `HF_TOKEN` does not exist in your Colab secrets.\n",
            "To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.\n",
            "You will be able to reuse this secret in all of your notebooks.\n",
            "Please note that authentication is recommended but still optional to access public models or datasets.\n",
            "  warnings.warn(\n",
            "Device set to use cuda:0\n",
            "2025-12-28 17:16:43.392 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:43.395 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:43.396 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:43.398 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:43.398 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:43.399 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:43.399 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:43.400 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:43.401 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:43.402 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:43.402 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:43.403 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:43.408 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:43.409 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:43.411 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:16:43.412 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ]
    }
  ]
}