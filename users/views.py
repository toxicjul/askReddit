from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json

from .langchain import get_agent_response
from .models import ChatMemory

from django.views.decorators.http import require_POST
from django.shortcuts import redirect
import sys


@login_required
def chat_view(request):
    """Render the chat page with history."""
    memory_obj, _ = ChatMemory.objects.get_or_create(user=request.user)
    history = memory_obj.memory  

    return render(request, "chat.html", {
        "history": history,
    })


@csrf_exempt
@login_required
def chat_api(request):
    """Handle fetch requests from the chat interface."""
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            
            message = data.get("message", "").strip()
            
            
            if not message:
                return JsonResponse({"error": "Empty message"}, status=400)

            response = get_agent_response(message, request.user)

            return JsonResponse({"reply": response})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)


@require_POST
@login_required
def reset_chat(request):
    ChatMemory.objects.filter(user=request.user).delete()
    return redirect('chat')  