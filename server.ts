import express from "express";
import cors from "cors";
import path from "path";
import fs from "fs";
import crypto from "crypto";
import multer from "multer";
import dotenv from "dotenv";
import { GoogleGenAI } from "@google/genai";

dotenv.config();

const app = express();
const PORT = parseInt(process.env.PORT || "3000", 10);
const HOST = process.env.HOST || "0.0.0.0";
const upload = multer({ limits: { fileSize: 50 * 1024 * 1024 } });

app.use(cors());
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));

// Initialize Gemini client (server-side only)
let genAIClient: GoogleGenAI | null = null;
function getGenAI(): GoogleGenAI | null {
  if (!genAIClient && process.env.GEMINI_API_KEY) {
    try {
      genAIClient = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
    } catch (err) {
      console.warn("Failed to initialize Google Gen AI client:", err);
    }
  }
  return genAIClient;
}

// In-memory data stores
interface Message {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp?: string;
  metadata?: Record<string, any>;
}

interface Session {
  id: string;
  name: string;
  model: string;
  endpoint_url: string;
  rag: any;
  archived: boolean;
  folder: string | null;
  total_tokens: number;
  is_important: boolean;
  created_at: string;
  updated_at: string;
  last_message_at: string;
  has_documents: boolean;
  has_images: boolean;
  mode: string;
  message_count: number;
  messages: Message[];
}

const initialSessionId = crypto.randomUUID();
const nowIso = new Date().toISOString();

const sessions: Map<string, Session> = new Map([
  [
    initialSessionId,
    {
      id: initialSessionId,
      name: "Welcome to Odysseus",
      model: "gemini-2.5-flash",
      endpoint_url: "/api/chat",
      rag: null,
      archived: false,
      folder: null,
      total_tokens: 0,
      is_important: false,
      created_at: nowIso,
      updated_at: nowIso,
      last_message_at: nowIso,
      has_documents: false,
      has_images: false,
      mode: "chat",
      message_count: 1,
      messages: [
        {
          role: "assistant",
          content:
            "Welcome to **Odysseus**! Your private AI workspace for chat, agents, notes, tasks, calendar, research, documents, and model workflows is running live in Google AI Studio. How can I assist you today?",
          timestamp: nowIso,
        },
      ],
    },
  ],
]);

const notes: Array<{ id: string; title: string; content: string; created_at: string; updated_at: string }> = [
  {
    id: "welcome-note",
    title: "Odysseus Quickstart",
    content: "# Welcome to Odysseus\n\nOdysseus provides integrated tools:\n- **Chat + Agents** with Gemini models\n- **Notes & Tasks** for personal knowledge\n- **Calendar** for schedule planning\n- **Documents & Research** editor",
    created_at: nowIso,
    updated_at: nowIso,
  },
];

const tasks: Array<{ id: string; title: string; completed: boolean; priority: string; due_date: string | null; created_at: string }> = [
  {
    id: "welcome-task-1",
    title: "Explore Odysseus features",
    completed: false,
    priority: "medium",
    due_date: null,
    created_at: nowIso,
  },
  {
    id: "welcome-task-2",
    title: "Try asking a question in Chat",
    completed: false,
    priority: "high",
    due_date: null,
    created_at: nowIso,
  },
];

const calendarEvents: Array<{ id: string; title: string; start: string; end: string; description: string; all_day?: boolean }> = [];

let appSettings = {
  auth_enabled: false,
  registration_enabled: false,
  default_model: "gemini-2.5-flash",
  default_endpoint_id: "gemini-cloud",
  disabled_tools: [],
  agent_email_confirm: true,
  tts_enabled: false,
  vision_enabled: true,
  theme: "one-dark",
};

const documents: Array<{ id: string; title: string; content: string; updated_at: string }> = [];
const memoryItems: Array<{ id: string; key: string; value: string; created_at: string }> = [];

// Persistence layer for Docker volume / local disk mounts
const DATA_DIR = process.env.DATA_DIR || path.join(process.cwd(), "data");
const DATA_FILE = path.join(DATA_DIR, "odysseus.json");

function loadPersistedData() {
  try {
    if (!fs.existsSync(DATA_DIR)) {
      fs.mkdirSync(DATA_DIR, { recursive: true });
    }
    if (fs.existsSync(DATA_FILE)) {
      const raw = fs.readFileSync(DATA_FILE, "utf-8");
      const data = JSON.parse(raw);
      if (Array.isArray(data.sessions) && data.sessions.length > 0) {
        sessions.clear();
        for (const s of data.sessions) {
          sessions.set(s.id, s);
        }
      }
      if (Array.isArray(data.notes) && data.notes.length > 0) {
        notes.length = 0;
        notes.push(...data.notes);
      }
      if (Array.isArray(data.tasks) && data.tasks.length > 0) {
        tasks.length = 0;
        tasks.push(...data.tasks);
      }
      if (Array.isArray(data.calendarEvents)) {
        calendarEvents.length = 0;
        calendarEvents.push(...data.calendarEvents);
      }
      if (Array.isArray(data.documents)) {
        documents.length = 0;
        documents.push(...data.documents);
      }
      if (Array.isArray(data.memoryItems)) {
        memoryItems.length = 0;
        memoryItems.push(...data.memoryItems);
      }
      if (data.settings && typeof data.settings === "object") {
        appSettings = { ...appSettings, ...data.settings };
      }
      console.log(`[Storage] Loaded persisted workspace data from ${DATA_FILE}`);
    }
  } catch (err) {
    console.warn("[Storage] Could not load persisted data, using defaults:", err);
  }
}

let saveTimer: NodeJS.Timeout | null = null;
function savePersistedData() {
  if (saveTimer) return;
  saveTimer = setTimeout(() => {
    saveTimer = null;
    try {
      if (!fs.existsSync(DATA_DIR)) {
        fs.mkdirSync(DATA_DIR, { recursive: true });
      }
      const data = {
        sessions: Array.from(sessions.values()),
        notes,
        tasks,
        calendarEvents,
        documents,
        memoryItems,
        settings: appSettings,
      };
      fs.writeFileSync(DATA_FILE, JSON.stringify(data, null, 2), "utf-8");
    } catch (err) {
      console.warn("[Storage] Failed to save persisted data:", err);
    }
  }, 250);
}

// Initial load from disk/volume if available
loadPersistedData();

// Helper: Serve HTML with CSP Nonce
function serveHtmlWithNonce(res: express.Response, filePath: string) {
  try {
    const fullPath = path.resolve(filePath);
    if (!fs.existsSync(fullPath)) {
      res.status(404).send("File not found");
      return;
    }
    let html = fs.readFileSync(fullPath, "utf-8");
    const nonce = crypto.randomBytes(16).toString("hex");
    html = html.replace(/\{\{CSP_NONCE\}\}/g, nonce);
    res.setHeader("Content-Type", "text/html; charset=utf-8");
    res.send(html);
  } catch (err) {
    console.error("Error serving HTML template:", err);
    res.status(500).send("Internal server error");
  }
}

// Static assets with explicit MIME types
app.use(
  "/static",
  express.static(path.join(process.cwd(), "static"), {
    setHeaders: (res, filePath) => {
      if (filePath.endsWith(".js") || filePath.endsWith(".mjs")) {
        res.setHeader("Content-Type", "application/javascript; charset=utf-8");
      } else if (filePath.endsWith(".css")) {
        res.setHeader("Content-Type", "text/css; charset=utf-8");
      } else if (filePath.endsWith(".woff2")) {
        res.setHeader("Content-Type", "font/woff2");
      }
    },
  })
);

// Serve manifest and favicon from static if requested at root
app.get("/manifest.json", (req, res) => {
  res.sendFile(path.join(process.cwd(), "static/manifest.json"));
});

app.get("/favicon.ico", (req, res) => {
  res.sendFile(path.join(process.cwd(), "static/icon.ico"));
});

// ==================== AUTH ROUTES ====================
app.get("/api/auth/status", (req, res) => {
  res.json({
    authenticated: true,
    username: "odysseus",
    role: "admin",
    is_admin: true,
    user: {
      username: "odysseus",
      is_admin: true,
    },
  });
});

app.get("/api/auth/settings", (req, res) => {
  res.json(appSettings);
});

app.post("/api/auth/settings", (req, res) => {
  appSettings = { ...appSettings, ...req.body };
  res.json(appSettings);
});

app.get("/api/auth/policy", (req, res) => {
  res.json({ require_admin: false });
});

app.get("/api/auth/features", (req, res) => {
  res.json({
    chat: true,
    notes: true,
    calendar: true,
    email: true,
    memory: true,
    gallery: true,
    tasks: true,
    library: true,
    cookbook: true,
  });
});

app.get("/api/auth/integrations/presets", (req, res) => {
  res.json([]);
});

app.post(["/api/auth/login", "/api/auth/logout", "/api/auth/setup", "/api/auth/signup"], (req, res) => {
  res.json({ ok: true });
});

// ==================== MODELS & PROVIDERS ====================
const geminiModels = [
  {
    endpoint_id: "gemini-cloud",
    endpoint_name: "Google Gemini",
    category: "api",
    url: "/api/chat",
    model_type: "llm",
    models: ["gemini-2.5-flash", "gemini-2.5-pro"],
    models_display: ["Gemini 2.5 Flash", "Gemini 2.5 Pro"],
    models_extra: ["gemini-2.0-flash", "gemini-1.5-flash"],
    models_extra_display: ["Gemini 2.0 Flash", "Gemini 1.5 Flash"],
    offline: false,
  },
];

app.get("/api/models", (req, res) => {
  res.json({
    items: geminiModels,
  });
});

app.get("/api/model-endpoints", (req, res) => {
  res.json([
    {
      id: "gemini-cloud",
      name: "Google Gemini",
      base_url: "https://generativelanguage.googleapis.com",
      provider: "google",
      is_enabled: true,
      category: "api",
      model_type: "llm",
    },
  ]);
});

app.get("/api/model-endpoints/probe-local", (req, res) => {
  res.json({});
});

app.get("/api/default-chat", (req, res) => {
  res.json({
    model: "gemini-2.5-flash",
    endpoint_id: "gemini-cloud",
    url: "/api/chat",
  });
});

// ==================== SESSIONS ====================
app.get("/api/sessions", (req, res) => {
  const activeSessions = Array.from(sessions.values())
    .filter((s) => !s.archived)
    .map((s) => ({
      id: s.id,
      name: s.name,
      model: s.model,
      endpoint_url: s.endpoint_url,
      rag: s.rag,
      archived: s.archived,
      folder: s.folder,
      total_tokens: s.total_tokens,
      is_important: s.is_important,
      created_at: s.created_at,
      updated_at: s.updated_at,
      last_message_at: s.last_message_at,
      has_documents: s.has_documents,
      has_images: s.has_images,
      mode: s.mode,
      message_count: s.messages.length,
    }));
  res.json(activeSessions);
});

app.get("/api/sessions/archived", (req, res) => {
  const archived = Array.from(sessions.values()).filter((s) => s.archived);
  res.json({
    sessions: archived,
    total: archived.length,
  });
});

app.post("/api/session", upload.any(), (req, res) => {
  const body = req.body || {};
  const id = crypto.randomUUID();
  const now = new Date().toISOString();
  const name = body.name || "New Chat";
  const model = body.model || "gemini-2.5-flash";
  const endpoint_url = body.endpoint_url || "/api/chat";

  const newSession: Session = {
    id,
    name,
    model,
    endpoint_url,
    rag: null,
    archived: false,
    folder: null,
    total_tokens: 0,
    is_important: false,
    created_at: now,
    updated_at: now,
    last_message_at: now,
    has_documents: false,
    has_images: false,
    mode: "chat",
    message_count: 0,
    messages: [],
  };

  sessions.set(id, newSession);
  res.json({
    id: newSession.id,
    name: newSession.name,
    model: newSession.model,
    endpoint_url: newSession.endpoint_url,
    created_at: newSession.created_at,
  });
});

app.get("/api/session/:id", (req, res) => {
  const session = sessions.get(req.params.id);
  if (!session) {
    res.status(404).json({ error: "Session not found" });
    return;
  }
  res.json(session);
});

app.patch("/api/session/:id", upload.any(), (req, res) => {
  const session = sessions.get(req.params.id);
  if (!session) {
    res.status(404).json({ error: "Session not found" });
    return;
  }
  const body = req.body || {};
  if (body.name !== undefined) session.name = body.name;
  if (body.folder !== undefined) session.folder = body.folder;
  if (body.model !== undefined) session.model = body.model;
  session.updated_at = new Date().toISOString();
  res.json(session);
});

app.delete("/api/session/:id", (req, res) => {
  sessions.delete(req.params.id);
  res.json({ ok: true });
});

app.post("/api/session/:id/archive", (req, res) => {
  const session = sessions.get(req.params.id);
  if (session) {
    session.archived = true;
  }
  res.json({ ok: true });
});

app.post("/api/session/:id/unarchive", (req, res) => {
  const session = sessions.get(req.params.id);
  if (session) {
    session.archived = false;
  }
  res.json({ ok: true });
});

app.post("/api/session/:id/important", upload.any(), (req, res) => {
  const session = sessions.get(req.params.id);
  if (session) {
    session.is_important = !session.is_important;
  }
  res.json({ ok: true });
});

app.post("/api/session/:id/compact", (req, res) => {
  res.json({ status: "compacted" });
});

app.get("/api/session/:id/context", (req, res) => {
  res.json({ total_tokens: 0 });
});

app.get("/api/session/:id/context_info", (req, res) => {
  const session = sessions.get(req.params.id);
  res.json({
    context_length: 1048576,
    model: session?.model || "gemini-2.5-flash",
  });
});

app.get("/api/history/:id", (req, res) => {
  const session = sessions.get(req.params.id);
  if (!session) {
    res.status(404).json({ error: "Session not found" });
    return;
  }
  const limit = req.query.limit ? parseInt(String(req.query.limit), 10) : 50;
  const offset = req.query.offset ? parseInt(String(req.query.offset), 10) : 0;
  const total = session.messages.length;

  res.json({
    history: session.messages,
    model: session.model,
    endpoint_url: session.endpoint_url,
    name: session.name,
    offset,
    limit,
    total,
    has_more_before: false,
    has_more_after: false,
  });
});

// ==================== CHAT STREAMING ====================
app.post("/api/chat_stream", upload.any(), async (req, res) => {
  const body = req.body || {};
  const messageText = (body.message || "").toString().trim();
  const sessionId = body.session || initialSessionId;
  const requestedModel = body.model || "gemini-2.5-flash";

  let session = sessions.get(sessionId);
  if (!session) {
    session = {
      id: sessionId,
      name: messageText.slice(0, 30) || "New Chat",
      model: requestedModel,
      endpoint_url: "/api/chat",
      rag: null,
      archived: false,
      folder: null,
      total_tokens: 0,
      is_important: false,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      last_message_at: new Date().toISOString(),
      has_documents: false,
      has_images: false,
      mode: "chat",
      message_count: 0,
      messages: [],
    };
    sessions.set(sessionId, session);
  }

  // Record user message
  const now = new Date().toISOString();
  if (messageText) {
    session.messages.push({
      role: "user",
      content: messageText,
      timestamp: now,
    });
    session.last_message_at = now;
    session.updated_at = now;
    if (session.messages.length === 1 || session.name === "New Chat") {
      session.name = messageText.slice(0, 32);
    }
  }

  // Set SSE streaming headers
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Odysseus-Run-Id", `run-${Date.now()}`);
  res.flushHeaders?.();

  const ai = getGenAI();

  let accumulatedReply = "";

  if (ai && process.env.GEMINI_API_KEY) {
    try {
      // Prepare conversation history for Gemini
      const contents = session.messages.map((m) => ({
        role: m.role === "assistant" ? "model" : "user",
        parts: [{ text: m.content }],
      }));

      // Pick model name
      const modelName = requestedModel.startsWith("gemini")
        ? requestedModel
        : "gemini-2.5-flash";

      const streamResult = await ai.models.generateContentStream({
        model: modelName,
        contents,
      });

      for await (const chunk of streamResult) {
        const delta = chunk.text || "";
        if (delta) {
          accumulatedReply += delta;
          res.write(`data: ${JSON.stringify({ delta })}\n\n`);
        }
      }
    } catch (err: any) {
      console.error("Gemini API stream error:", err);
      const errMessage = `\n\n*(Gemini API error: ${err?.message || "Failed to generate response"})*`;
      accumulatedReply += errMessage;
      res.write(`data: ${JSON.stringify({ delta: errMessage })}\n\n`);
    }
  } else {
    // Graceful fallback when GEMINI_API_KEY is not yet populated
    const fallbackText =
      `Hello! I received your message: "${messageText}".\n\n` +
      `Odysseus is running successfully in AI Studio. To enable full Gemini model intelligence, make sure \`GEMINI_API_KEY\` is configured in the environment or Settings.`;

    const words = fallbackText.split(" ");
    for (const word of words) {
      const delta = word + " ";
      accumulatedReply += delta;
      res.write(`data: ${JSON.stringify({ delta })}\n\n`);
      await new Promise((r) => setTimeout(r, 25));
    }
  }

  // Record assistant message
  session.messages.push({
    role: "assistant",
    content: accumulatedReply,
    timestamp: new Date().toISOString(),
  });
  session.last_message_at = new Date().toISOString();
  session.message_count = session.messages.length;

  res.write("data: [DONE]\n\n");
  res.end();
});

app.get("/api/chat/stream_status/:id", (req, res) => {
  res.json({ status: "idle" });
});

app.post("/api/chat/stop/:id", (req, res) => {
  res.json({ ok: true });
});

// ==================== NOTES ====================
app.get("/api/notes", (req, res) => {
  res.json(notes);
});

app.post("/api/notes", (req, res) => {
  const { title = "Untitled Note", content = "" } = req.body || {};
  const note = {
    id: crypto.randomUUID(),
    title,
    content,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };
  notes.unshift(note);
  res.json(note);
});

app.put("/api/notes/:id", (req, res) => {
  const note = notes.find((n) => n.id === req.params.id);
  if (!note) {
    res.status(404).json({ error: "Note not found" });
    return;
  }
  const { title, content } = req.body || {};
  if (title !== undefined) note.title = title;
  if (content !== undefined) note.content = content;
  note.updated_at = new Date().toISOString();
  res.json(note);
});

app.delete("/api/notes/:id", (req, res) => {
  const index = notes.findIndex((n) => n.id === req.params.id);
  if (index !== -1) notes.splice(index, 1);
  res.json({ ok: true });
});

// ==================== TASKS ====================
app.get("/api/tasks", (req, res) => {
  res.json(tasks);
});

app.post("/api/tasks", (req, res) => {
  const { title = "New Task", priority = "medium", due_date = null } = req.body || {};
  const task = {
    id: crypto.randomUUID(),
    title,
    completed: false,
    priority,
    due_date,
    created_at: new Date().toISOString(),
  };
  tasks.unshift(task);
  res.json(task);
});

app.patch("/api/tasks/:id", (req, res) => {
  const task = tasks.find((t) => t.id === req.params.id);
  if (!task) {
    res.status(404).json({ error: "Task not found" });
    return;
  }
  const { title, completed, priority, due_date } = req.body || {};
  if (title !== undefined) task.title = title;
  if (completed !== undefined) task.completed = completed;
  if (priority !== undefined) task.priority = priority;
  if (due_date !== undefined) task.due_date = due_date;
  res.json(task);
});

app.delete("/api/tasks/:id", (req, res) => {
  const index = tasks.findIndex((t) => t.id === req.params.id);
  if (index !== -1) tasks.splice(index, 1);
  res.json({ ok: true });
});

// ==================== CALENDAR ====================
app.get("/api/calendar/events", (req, res) => {
  res.json(calendarEvents);
});

app.get("/api/calendar/calendars", (req, res) => {
  res.json([{ id: "default", name: "Personal Calendar", color: "#e06c75" }]);
});

app.post("/api/calendar/events", (req, res) => {
  const { title = "New Event", start = "", end = "", description = "", all_day = false } = req.body || {};
  const event = {
    id: crypto.randomUUID(),
    title,
    start,
    end,
    description,
    all_day,
  };
  calendarEvents.push(event);
  res.json(event);
});

app.delete("/api/calendar/events/:id", (req, res) => {
  const index = calendarEvents.findIndex((e) => e.id === req.params.id);
  if (index !== -1) calendarEvents.splice(index, 1);
  res.json({ ok: true });
});

// ==================== DOCUMENTS & LIBRARY ====================
app.get("/api/documents/library", (req, res) => {
  res.json({
    documents,
    total: documents.length,
  });
});

app.post("/api/document", (req, res) => {
  const { title = "Untitled", content = "" } = req.body || {};
  const doc = {
    id: crypto.randomUUID(),
    title,
    content,
    updated_at: new Date().toISOString(),
  };
  documents.unshift(doc);
  res.json(doc);
});

// ==================== MEMORY, PRESETS, GALLERY, RESEARCH, EMAIL ====================
app.get("/api/memory", (req, res) => {
  res.json(memoryItems);
});

app.get("/api/presets", (req, res) => {
  res.json([]);
});

app.get("/api/presets/templates", (req, res) => {
  res.json([]);
});

app.get("/api/presets/groups", (req, res) => {
  res.json([]);
});

app.get("/api/gallery/library", (req, res) => {
  res.json({ images: [], total: 0 });
});

app.get("/api/research/library", (req, res) => {
  res.json({ items: [], total: 0 });
});

app.get("/api/email/accounts", (req, res) => {
  res.json([]);
});

app.get("/api/email/list", (req, res) => {
  res.json([]);
});

app.get("/api/email/unread-state", (req, res) => {
  res.json({ unread: 0 });
});

app.get("/api/tools", (req, res) => {
  res.json([]);
});

app.get("/api/mcp/servers", (req, res) => {
  res.json([]);
});

app.get("/api/prefs/theme", (req, res) => {
  res.json({ theme: "one-dark" });
});

app.get("/api/prefs/custom-themes", (req, res) => {
  res.json([]);
});

app.get("/api/tokens", (req, res) => {
  res.json([]);
});

app.get("/api/webhooks", (req, res) => {
  res.json([]);
});

app.get("/api/cookbook/state", (req, res) => {
  res.json({});
});

app.get("/api/cookbook/packages", (req, res) => {
  res.json({});
});

app.get("/api/cookbook/gpus", (req, res) => {
  res.json([]);
});

app.get("/api/stt/stats", (req, res) => {
  res.json({});
});

app.get("/api/tts/stats", (req, res) => {
  res.json({});
});

app.get("/api/version", (req, res) => {
  res.json({ version: "1.0.0", commit: "ai-studio" });
});

app.get("/api/runtime", (req, res) => {
  res.json({
    platform: process.platform,
    arch: process.arch,
    memory: process.memoryUsage(),
    uptime: process.uptime(),
  });
});

app.get("/api/activity/heartbeat", (req, res) => {
  res.json({ status: "ok" });
});

// Fallback for any unmapped API endpoint
app.all("/api/{*path}", (req, res) => {
  if (req.method === "GET") {
    res.json([]);
  } else {
    res.json({ ok: true });
  }
});

// ==================== APP PAGES & DEEP LINKS ====================
app.get(
  [
    "/",
    "/notes",
    "/calendar",
    "/cookbook",
    "/email",
    "/memory",
    "/gallery",
    "/tasks",
    "/library",
  ],
  (req, res) => {
    serveHtmlWithNonce(res, path.join(process.cwd(), "static/index.html"));
  }
);

app.get("/login", (req, res) => {
  if (!appSettings.auth_enabled) {
    res.redirect("/");
    return;
  }
  serveHtmlWithNonce(res, path.join(process.cwd(), "static/login.html"));
});

app.get("/backgrounds", (req, res) => {
  const bgPath = path.join(process.cwd(), "static/backgrounds.html");
  if (fs.existsSync(bgPath)) {
    serveHtmlWithNonce(res, bgPath);
  } else {
    serveHtmlWithNonce(res, path.join(process.cwd(), "static/index.html"));
  }
});

// SPA fallback for any unmatched client route
app.use((req, res) => {
  serveHtmlWithNonce(res, path.join(process.cwd(), "static/index.html"));
});

app.listen(PORT, HOST, () => {
  console.log(`Odysseus server running on http://${HOST}:${PORT}`);
});
