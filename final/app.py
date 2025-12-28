{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "authorship_tag": "ABX9TyPDm+2l0zxbLFx9cANJx2nN",
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
        "st.set_page_config(page_title=\"GenAI Studio\", layout=\"wide\")\n",
        "st.title(\"🎨 GenAI Art & Chat Studio\")\n",
        "\n",
        "@st.cache_resource\n",
        "def load_chat_model():\n",
        "    return pipeline(\"text2text-generation\", model=\"google/flan-t5-base\")\n",
        "\n",
        "@st.cache_resource\n",
        "def load_art_model():\n",
        "    model_id = \"runwayml/stable-diffusion-v1-5\"\n",
        "    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)\n",
        "    return pipe\n",
        "\n",
        "mode = st.sidebar.radio(\"Select Mode:\", [\"Chat\", \"Art\"])\n",
        "\n",
        "if mode == \"Chat\":\n",
        "    st.subheader(\"💬 AI Conversationalist\")\n",
        "    chat_model = load_chat_model()\n",
        "    user_input = st.text_input(\"Message:\")\n",
        "    if st.button(\"Send\") and user_input:\n",
        "        response = chat_model(user_input, max_length=100)\n",
        "        st.write(f\"**AI:** {response[0]['generated_text']}\")\n",
        "\n",
        "else:\n",
        "    st.subheader(\"🖼️ Creative Image Engine\")\n",
        "    prompt = st.text_input(\"Prompt:\", \"A cinematic landscape, ultra high definition\")\n",
        "\n",
        "    if st.button(\"Generate\"):\n",
        "        with st.spinner(\"Processing...\"):\n",
        "            art_model = load_art_model()\n",
        "            image = art_model(prompt).images[0]\n",
        "            st.session_state['current_img'] = image\n",
        "\n",
        "    if 'current_img' in st.session_state:\n",
        "        st.image(st.session_state['current_img'], caption=\"Original\")\n",
        "\n",
        "        filter_type = st.selectbox(\"Apply Filter:\", [\"None\", \"Grayscale\", \"Blur\", \"Edges\"])\n",
        "\n",
        "        img = st.session_state['current_img']\n",
        "        if filter_type == \"Grayscale\":\n",
        "            img = ImageOps.grayscale(img)\n",
        "        elif filter_type == \"Blur\":\n",
        "            img = img.filter(ImageFilter.BLUR)\n",
        "        elif filter_type == \"Edges\":\n",
        "            img = img.filter(ImageFilter.FIND_EDGES)\n",
        "\n",
        "        st.image(img, caption=f\"Result: {filter_type}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "L9WMXYgz_8Ol",
        "outputId": "48bcce04-d2e7-4337-81a4-1ae37a6718cd"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Flax classes are deprecated and will be removed in Diffusers v1.0.0. We recommend migrating to PyTorch classes or pinning your version of Diffusers.\n",
            "Flax classes are deprecated and will be removed in Diffusers v1.0.0. We recommend migrating to PyTorch classes or pinning your version of Diffusers.\n",
            "2025-12-28 17:27:41.719 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:41.724 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:41.816 \n",
            "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
            "  command:\n",
            "\n",
            "    streamlit run /usr/local/lib/python3.12/dist-packages/colab_kernel_launcher.py [ARGUMENTS]\n",
            "2025-12-28 17:27:41.817 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:41.819 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:41.823 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:41.824 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:41.826 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:41.829 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:41.829 Session state does not function when running a script without `streamlit run`\n",
            "2025-12-28 17:27:41.831 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:41.833 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:41.834 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:41.837 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:41.839 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:41.841 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:41.843 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:41.843 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:41.844 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:41.844 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:42.348 Thread 'Thread-3': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:42.349 Thread 'Thread-3': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:42.349 Thread 'Thread-3': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: \n",
            "The secret `HF_TOKEN` does not exist in your Colab secrets.\n",
            "To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.\n",
            "You will be able to reuse this secret in all of your notebooks.\n",
            "Please note that authentication is recommended but still optional to access public models or datasets.\n",
            "  warnings.warn(\n",
            "Device set to use cuda:0\n",
            "2025-12-28 17:27:46.046 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:46.047 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:46.048 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:46.049 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:46.049 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:46.050 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:46.051 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:46.052 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:46.053 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:46.054 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:46.054 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:46.055 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:46.056 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:46.057 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:46.057 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-28 17:27:46.058 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ]
    }
  ]
}