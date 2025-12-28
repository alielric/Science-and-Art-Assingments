{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "authorship_tag": "ABX9TyMYzA1Lwck5mxTzv21m1wxq",
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
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "l9VTff-SzHsK",
        "outputId": "d413e457-6f0c-4f58-d48e-98aa30456510"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "  Installing build dependencies ... \u001b[?25l\u001b[?25hdone\n",
            "  Getting requirements to build wheel ... \u001b[?25l\u001b[?25hdone\n",
            "  Preparing metadata (pyproject.toml) ... \u001b[?25l\u001b[?25hdone\n"
          ]
        }
      ],
      "source": [
        "!pip install -q streamlit transformers torch diffusers pillow\n",
        "!pip install -q git+https://github.com/huggingface/diffusers\n",
        "!pip install -q pyngrok"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "%%writefile app.py\n",
        "import streamlit as st\n",
        "from transformers import pipeline\n",
        "import torch\n",
        "from diffusers import StableDiffusionPipeline\n",
        "from PIL import Image, ImageOps, ImageFilter\n",
        "import io\n",
        "\n",
        "st.set_page_config(page_title=\"GenAI Art & Chat Studio\", layout=\"wide\")\n",
        "\n",
        "# --- Başlık ve Model Yükleme ---\n",
        "st.title(\"🎨 GenAI Art & Chat Studio (Week 12 Final)\")\n",
        "st.sidebar.header(\"Ayarlar\")\n",
        "\n",
        "mode = st.sidebar.radio(\"Mod Seçin:\", [\"Chat Mode\", \"Art Mode\"])\n",
        "\n",
        "@st.cache_resource\n",
        "def load_chat_model():\n",
        "    # Ücretsiz ve hafif bir model (Week 5 & 9 uyumlu)\n",
        "    return pipeline(\"text2text-generation\", model=\"google/flan-t5-base\")\n",
        "\n",
        "@st.cache_resource\n",
        "def load_art_model():\n",
        "    # Stable Diffusion (Week 7 & 9 uyumlu)\n",
        "    model_id = \"runwayml/stable-diffusion-v1-5\"\n",
        "    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)\n",
        "    if torch.cuda.is_available():\n",
        "        pipe = pipe.to(\"cuda\")\n",
        "    return pipe\n",
        "\n",
        "# --- Chat Mode ---\n",
        "if mode == \"Chat Mode\":\n",
        "    st.subheader(\"💬 AI Chatbot\")\n",
        "    chat_model = load_chat_model()\n",
        "    user_input = st.text_input(\"Bir şey sorun:\")\n",
        "    if st.button(\"Gönder\") and user_input:\n",
        "        response = chat_model(user_input, max_length=100)\n",
        "        st.write(\"**AI:**\", response[0]['generated_text'])\n",
        "\n",
        "# --- Art Mode ---\n",
        "else:\n",
        "    st.subheader(\"🖼️ Generative Art & Filters\")\n",
        "    art_model = load_art_model()\n",
        "\n",
        "    prompt = st.text_input(\"Görsel açıklaması girin (İngilizce):\", \"A futuristic city in watercolor style\")\n",
        "\n",
        "    col1, col2 = st.columns(2)\n",
        "\n",
        "    with col1:\n",
        "        if st.button(\"Görsel Oluştur\"):\n",
        "            with st.spinner(\"Görsel oluşturuluyor...\"):\n",
        "                image = art_model(prompt).images[0]\n",
        "                st.session_state['current_img'] = image\n",
        "                st.image(image, caption=\"Oluşturulan Görsel\")\n",
        "\n",
        "    # --- Filtreleme (Week 11 - Artistic Style Enhancement) ---\n",
        "    if 'current_img' in st.session_state:\n",
        "        with col2:\n",
        "            st.write(\"### Artistic Filters\")\n",
        "            filter_type = st.selectbox(\"Filtre Seçin:\", [\"Orijinal\", \"Siyah Beyaz\", \"Blur (Bulanık)\", \"Kenar Belirleme\"])\n",
        "\n",
        "            img = st.session_state['current_img']\n",
        "            if filter_type == \"Siyah Beyaz\":\n",
        "                img = ImageOps.grayscale(img)\n",
        "            elif filter_type == \"Blur (Bulanık)\":\n",
        "                img = img.filter(ImageFilter.BLUR)\n",
        "            elif filter_type == \"Kenar Belirleme\":\n",
        "                img = img.filter(ImageFilter.FIND_EDGES)\n",
        "\n",
        "            st.image(img, caption=f\"Filtre: {filter_type}\")\n",
        "\n",
        "            # Kaydetme butonu\n",
        "            buf = io.BytesIO()\n",
        "            img.save(buf, format=\"PNG\")\n",
        "            st.download_button(\"Görseli İndir\", buf.getvalue(), \"ai_art.png\", \"image/png\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "r8jvLCvd0ATT",
        "outputId": "3c427d9c-d7ab-4bcb-bb96-013362be77f1"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Overwriting app.py\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "!streamlit run app.py & npx localtunnel --port 8501"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BD8yAbgx0CEE",
        "outputId": "096981da-5cae-4acf-eeac-72cbc3ad4128"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[1G\u001b[0K⠙\u001b[1G\u001b[0K⠹\u001b[1G\u001b[0K⠸\u001b[1G\u001b[0K⠼\u001b[1G\u001b[0K⠴\n",
            "Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.\n",
            "\u001b[0m\n",
            "\u001b[1G\u001b[0K⠦\u001b[1G\u001b[0K⠧\u001b[1G\u001b[0K⠇\u001b[1G\u001b[0K⠏\u001b[1G\u001b[0K⠋\u001b[1G\u001b[0K⠙\u001b[1G\u001b[0K⠹\u001b[1G\u001b[0K⠸\u001b[1G\u001b[0K⠼\u001b[1G\u001b[0K⠴\u001b[1G\u001b[0K⠦\u001b[1G\u001b[0K⠧\u001b[1G\u001b[0K⠇\u001b[1G\u001b[0K⠏\u001b[1G\u001b[0K⠋\u001b[1G\u001b[0K\u001b[0m\n",
            "\u001b[34m\u001b[1m  You can now view your Streamlit app in your browser.\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Local URL: \u001b[0m\u001b[1mhttp://localhost:8501\u001b[0m\n",
            "\u001b[34m  Network URL: \u001b[0m\u001b[1mhttp://172.28.0.12:8501\u001b[0m\n",
            "\u001b[34m  External URL: \u001b[0m\u001b[1mhttp://34.87.9.225:8501\u001b[0m\n",
            "\u001b[0m\n",
            "your url is: https://crazy-nights-sink.loca.lt\n",
            "2025-12-28 16:56:14.520611: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered\n",
            "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR\n",
            "E0000 00:00:1766940974.542509    7171 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered\n",
            "E0000 00:00:1766940974.549388    7171 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered\n",
            "W0000 00:00:1766940974.566689    7171 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.\n",
            "W0000 00:00:1766940974.566716    7171 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.\n",
            "W0000 00:00:1766940974.566720    7171 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.\n",
            "W0000 00:00:1766940974.566727    7171 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.\n",
            "2025-12-28 16:56:14.571599: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.\n",
            "To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.\n",
            "Flax classes are deprecated and will be removed in Diffusers v1.0.0. We recommend migrating to PyTorch classes or pinning your version of Diffusers.\n",
            "Flax classes are deprecated and will be removed in Diffusers v1.0.0. We recommend migrating to PyTorch classes or pinning your version of Diffusers.\n",
            "Device set to use cuda:0\n"
          ]
        }
      ]
    }
  ]
}