from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import ValidationError

from app.models import ChatForm, Conversation, ConversationSettings, VectorStoreType
from app.services.conversation_store import ConversationStore
from app.services.ollama_service import OllamaModelService
from app.services.rag_search import RagSearchService

router = APIRouter()

CHUNK_COUNT_CHOICES = [10, 25, 50, 100]
SESSION_KEY = "conversation_id"


def _default_settings(request: Request) -> ConversationSettings:
    ollama_model_service: OllamaModelService = request.app.state.ollama_model_service
    settings = request.app.state.settings
    return ConversationSettings(
        chat_model=ollama_model_service.default_chat_model(),
        vector_store=settings.default_vector_store,
        chunk_count=settings.search_top_k,
        ticker="",
        form="",
    )


def _get_conversation(request: Request) -> Conversation | None:
    store: ConversationStore = request.app.state.conversation_store
    conversation_id = request.session.get(SESSION_KEY)
    return store.get(conversation_id)


def _save_conversation(request: Request, conversation: Conversation) -> None:
    store: ConversationStore = request.app.state.conversation_store
    store.save(conversation)
    request.session[SESSION_KEY] = conversation.id


def _clear_conversation(request: Request) -> None:
    store: ConversationStore = request.app.state.conversation_store
    conversation_id = request.session.pop(SESSION_KEY, None)
    if conversation_id:
        store.delete(conversation_id)


def _render_chat(
    request: Request,
    conversation: Conversation | None,
    settings: ConversationSettings,
    field_errors: dict[str, str] | None = None,
) -> HTMLResponse:
    ollama_model_service: OllamaModelService = request.app.state.ollama_model_service
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "conversation": conversation,
            "settings": settings,
            "chat_models": ollama_model_service.list_chat_models(),
            "vector_stores": [store.value for store in VectorStoreType],
            "chunk_count_choices": CHUNK_COUNT_CHOICES,
            "field_errors": field_errors or {},
        },
    )


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    conversation = _get_conversation(request)
    settings = conversation.settings if conversation else _default_settings(request)
    return _render_chat(request, conversation, settings)


@router.post("/chat", response_class=HTMLResponse)
def chat(
    request: Request,
    message: str = Form(""),
    chat_model: str = Form(""),
    vector_store: str = Form(""),
    chunk_count: int = Form(10),
    ticker: str = Form(""),
    form: str = Form(""),
) -> HTMLResponse:
    ollama_model_service: OllamaModelService = request.app.state.ollama_model_service
    rag_search_service: RagSearchService = request.app.state.rag_search_service
    default_settings = _default_settings(request)

    field_errors: dict[str, str] = {}
    try:
        chat_form = ChatForm(
            message=message,
            chat_model=chat_model or default_settings.chat_model,
            vector_store=vector_store or default_settings.vector_store,
            chunk_count=chunk_count,
            ticker=ticker,
            form=form,
        )
    except ValidationError as exc:
        settings = ConversationSettings(
            chat_model=chat_model or default_settings.chat_model,
            vector_store=vector_store or default_settings.vector_store,
            chunk_count=chunk_count if 1 <= chunk_count <= 500 else default_settings.chunk_count,
            ticker=ticker,
            form=form,
        )
        for error in exc.errors():
            loc = error.get("loc", ())
            if loc:
                field_errors[str(loc[0])] = error.get("msg", "Invalid value.")
        return _render_chat(request, _get_conversation(request), settings, field_errors)

    if not ollama_model_service.is_known_chat_model(chat_form.chat_model):
        field_errors["chat_model"] = "Select a valid Ollama model."

    try:
        chat_form.to_settings().vector_store_type()
    except ValueError:
        field_errors["vector_store"] = "Select a valid vector store."

    if field_errors:
        return _render_chat(request, _get_conversation(request), chat_form.to_settings(), field_errors)

    conversation = _get_conversation(request)
    if conversation is None:
        conversation = request.app.state.conversation_store.create(chat_form.to_settings())
    else:
        conversation = conversation.model_copy(update={"settings": chat_form.to_settings()})

    conversation = rag_search_service.continue_conversation(conversation, chat_form.message)
    _save_conversation(request, conversation)
    return _render_chat(request, conversation, conversation.settings)


@router.post("/chat/new")
def new_chat(request: Request) -> RedirectResponse:
    _clear_conversation(request)
    return RedirectResponse(url="/", status_code=303)


@router.post("/search", response_class=HTMLResponse, include_in_schema=False)
def search_legacy(
    request: Request,
    question: str = Form(""),
    chat_model: str = Form(""),
    vector_store: str = Form(""),
    chunk_count: int = Form(10),
    ticker: str = Form(""),
    form: str = Form(""),
) -> HTMLResponse:
    """Backward-compatible alias that starts or continues a conversation."""
    return chat(
        request,
        message=question,
        chat_model=chat_model,
        vector_store=vector_store,
        chunk_count=chunk_count,
        ticker=ticker,
        form=form,
    )
