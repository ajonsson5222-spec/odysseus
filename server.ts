import express from "express";
import cors from "cors";
import path from "path";
import fs from "fs";
import crypto from "crypto";
import multer from "multer";
import dotenv from "dotenv";
import { GoogleGenAI } from "@google/genai";

dotenv.config();

// Also attempt loading .env from data directory if mounted
const DATA_DIR = process.env.DATA_DIR || path.join(process.cwd(), "data");
const DATA_FILE = path.join(DATA_DIR, "odysseus.json");
try {
  const dataEnvPath = path.join(DATA_DIR, ".env");
  if (fs.existsSync(dataEnvPath)) {
    dotenv.config({ path: dataEnvPath, override: false });
  }
} catch (_) {}

const app = express();
const PORT = parseInt(process.env.PORT || "3000", 10);
const HOST = process.env.HOST || "0.0.0.0";
const upload = multer({ limits: { fileSize: 50 * 1024 * 1024 } });

app.use(cors());
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));

// Forward-declare model endpoints for dynamic key lookups
let genAIClient: GoogleGenAI | null = null;
let configuredGeminiKey: string | null = null;

function getActiveGeminiApiKey(): string | null {
  if (process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY.trim()) {
    return process.env.GEMINI_API_KEY.trim();
  }
  // Check endpoints array if initialized
  if (typeof modelEndpoints !== "undefined" && Array.isArray(modelEndpoints)) {
    const ep = modelEndpoints.find(
      (e) => (e.provider === "gemini" || e.provider === "google" || e.id === "gemini-cloud" || e.base_url?.includes("generativelanguage.googleapis.com")) && e.api_key && e.api_key.trim()
    );
    if (ep && ep.api_key) return ep.api_key.trim();
    const anyEp = modelEndpoints.find((e) => e.api_key && e.api_key.trim().startsWith("AIza"));
    if (anyEp && anyEp.api_key) return anyEp.api_key.trim();
  }
  // Check appSettings if initialized
  if (typeof appSettings !== "undefined" && (appSettings as any)?.gemini_api_key && typeof (appSettings as any).gemini_api_key === "string") {
    return (appSettings as any).gemini_api_key.trim();
  }
  // Check DATA_DIR/.env directly
  try {
    const dataEnv = path.join(DATA_DIR, ".env");
    if (fs.existsSync(dataEnv)) {
      const parsed = dotenv.parse(fs.readFileSync(dataEnv, "utf-8"));
      if (parsed.GEMINI_API_KEY && parsed.GEMINI_API_KEY.trim()) {
        process.env.GEMINI_API_KEY = parsed.GEMINI_API_KEY.trim();
        return parsed.GEMINI_API_KEY.trim();
      }
    }
  } catch (_) {}
  return null;
}

function getGenAI(): GoogleGenAI | null {
  const activeKey = getActiveGeminiApiKey();
  if (!activeKey) return null;
  if (!genAIClient || configuredGeminiKey !== activeKey) {
    try {
      genAIClient = new GoogleGenAI({ apiKey: activeKey });
      configuredGeminiKey = activeKey;
    } catch (err) {
      console.warn("Failed to initialize Google Gen AI client:", err);
      return null;
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

interface DocumentItem {
  id: string;
  session_id?: string;
  session_name?: string;
  title: string;
  content: string;
  current_content?: string;
  language?: string;
  preview?: string;
  version_count?: number;
  is_active?: boolean;
  archived?: boolean;
  created_at?: string;
  updated_at: string;
  source_email_uid?: string | null;
  source_email_folder?: string | null;
  source_email_account_id?: string | null;
  source_email_message_id?: string | null;
}

const documents: Array<DocumentItem> = [];
const memoryItems: Array<{ id: string; key: string; value: string; created_at: string }> = [];

export interface ModelEndpoint {
  id: string;
  name: string;
  base_url: string;
  api_key?: string;
  api_key_fingerprint?: string;
  has_key?: boolean;
  provider?: string;
  is_enabled: boolean;
  online: boolean;
  category: "local" | "api";
  model_type: "llm" | "image";
  models?: string[];
  models_display?: string[];
  models_extra?: string[];
  models_extra_display?: string[];
  model_count?: number;
  status?: string;
  ping_error?: string | null;
}

const defaultModelEndpoints: ModelEndpoint[] = [
  {
    id: "gemini-cloud",
    name: "Google Gemini",
    base_url: "https://generativelanguage.googleapis.com",
    provider: "google",
    is_enabled: true,
    online: true,
    category: "api",
    model_type: "llm",
    models: ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-flash"],
    models_display: ["Gemini 2.5 Flash", "Gemini 2.5 Pro", "Gemini 2.0 Flash", "Gemini 1.5 Flash"],
    model_count: 4,
    has_key: Boolean(process.env.GEMINI_API_KEY),
  },
];

if (process.env.GROQ_API_KEY) {
  defaultModelEndpoints.push({
    id: "groq-cloud",
    name: "Groq",
    base_url: "https://api.groq.com/openai/v1",
    api_key: process.env.GROQ_API_KEY,
    api_key_fingerprint: process.env.GROQ_API_KEY.slice(0, 4) + "...",
    has_key: true,
    provider: "groq",
    is_enabled: true,
    online: true,
    category: "api",
    model_type: "llm",
    models: ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    models_display: ["Llama 3.3 70B", "Llama 3.1 8B", "Mixtral 8x7B"],
    model_count: 3,
  });
}

if (process.env.OPENAI_API_KEY) {
  defaultModelEndpoints.push({
    id: "openai-cloud",
    name: "OpenAI",
    base_url: "https://api.openai.com/v1",
    api_key: process.env.OPENAI_API_KEY,
    api_key_fingerprint: process.env.OPENAI_API_KEY.slice(0, 4) + "...",
    has_key: true,
    provider: "openai",
    is_enabled: true,
    online: true,
    category: "api",
    model_type: "llm",
    models: ["gpt-4o", "gpt-4o-mini", "o3-mini"],
    models_display: ["GPT-4o", "GPT-4o Mini", "o3 Mini"],
    model_count: 3,
  });
}

if (process.env.OLLAMA_HOST || process.env.LLM_HOST) {
  const host = process.env.OLLAMA_HOST || process.env.LLM_HOST || "http://localhost:11434";
  const baseUrl = host.endsWith("/v1") ? host : `${host.replace(/\/+$/, "")}/v1`;
  defaultModelEndpoints.push({
    id: "ollama-local",
    name: "Ollama (Local)",
    base_url: baseUrl,
    has_key: false,
    provider: "ollama",
    is_enabled: true,
    online: true,
    category: "local",
    model_type: "llm",
    models: ["llama3", "mistral", "qwen2.5"],
    models_display: ["Llama 3", "Mistral", "Qwen 2.5"],
    model_count: 3,
  });
}

const modelEndpoints: Array<ModelEndpoint> = [...defaultModelEndpoints];

function setActiveGeminiApiKey(key: string) {
  const cleanKey = (key || "").trim();
  if (!cleanKey) return;
  process.env.GEMINI_API_KEY = cleanKey;
  (appSettings as any).gemini_api_key = cleanKey;
  genAIClient = null; // force reinit on next use
  configuredGeminiKey = null;

  let geminiEp = modelEndpoints.find((e) => e.id === "gemini-cloud" || e.provider === "gemini" || e.provider === "google");
  if (!geminiEp) {
    geminiEp = {
      id: "gemini-cloud",
      name: "Google Gemini",
      base_url: "https://generativelanguage.googleapis.com/v1beta/openai",
      provider: "gemini",
      is_enabled: true,
      online: true,
      category: "api",
      model_type: "llm",
      models: ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-flash"],
      models_display: ["Gemini 2.5 Flash", "Gemini 2.5 Pro", "Gemini 2.0 Flash", "Gemini 1.5 Flash"],
      model_count: 4,
      has_key: true,
      api_key: cleanKey,
      api_key_fingerprint: cleanKey.slice(0, 4) + "..." + cleanKey.slice(-4),
    };
    modelEndpoints.unshift(geminiEp);
  } else {
    geminiEp.api_key = cleanKey;
    geminiEp.api_key_fingerprint = cleanKey.slice(0, 4) + "..." + cleanKey.slice(-4);
    geminiEp.has_key = true;
    geminiEp.online = true;
    geminiEp.models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-flash"];
    geminiEp.models_display = ["Gemini 2.5 Flash", "Gemini 2.5 Pro", "Gemini 2.0 Flash", "Gemini 1.5 Flash"];
    geminiEp.model_count = 4;
  }

  // Also write to DATA_DIR/.env so host mounts stay persistent and easy to inspect
  try {
    const dataEnv = path.join(DATA_DIR, ".env");
    let currentContent = fs.existsSync(dataEnv) ? fs.readFileSync(dataEnv, "utf-8") : "";
    if (/^GEMINI_API_KEY=/m.test(currentContent)) {
      currentContent = currentContent.replace(/^GEMINI_API_KEY=.*$/m, `GEMINI_API_KEY=${cleanKey}`);
    } else {
      currentContent = (currentContent ? currentContent.trimEnd() + "\n" : "") + `GEMINI_API_KEY=${cleanKey}\n`;
    }
    fs.writeFileSync(dataEnv, currentContent, "utf-8");
  } catch (_) {}

  savePersistedData();
  console.log("[Auth] Active Gemini API Key successfully saved and enabled.");
}

function setActiveGroqApiKey(key: string) {
  const cleanKey = (key || "").trim();
  if (!cleanKey) return;
  process.env.GROQ_API_KEY = cleanKey;
  (appSettings as any).groq_api_key = cleanKey;

  let groqEp = modelEndpoints.find((e) => e.id === "groq-cloud" || e.provider === "groq");
  if (!groqEp) {
    groqEp = {
      id: "groq-cloud",
      name: "Groq",
      base_url: "https://api.groq.com/openai/v1",
      api_key: cleanKey,
      api_key_fingerprint: cleanKey.slice(0, 4) + "...",
      has_key: true,
      provider: "groq",
      is_enabled: true,
      online: true,
      category: "api",
      model_type: "llm",
      models: ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
      models_display: ["Llama 3.3 70B", "Llama 3.1 8B", "Mixtral 8x7B"],
      model_count: 3,
    };
    modelEndpoints.push(groqEp);
  } else {
    groqEp.api_key = cleanKey;
    groqEp.api_key_fingerprint = cleanKey.slice(0, 4) + "...";
    groqEp.has_key = true;
    groqEp.online = true;
  }

  try {
    const dataEnv = path.join(DATA_DIR, ".env");
    let currentContent = fs.existsSync(dataEnv) ? fs.readFileSync(dataEnv, "utf-8") : "";
    if (/^GROQ_API_KEY=/m.test(currentContent)) {
      currentContent = currentContent.replace(/^GROQ_API_KEY=.*$/m, `GROQ_API_KEY=${cleanKey}`);
    } else {
      currentContent = (currentContent ? currentContent.trimEnd() + "\n" : "") + `GROQ_API_KEY=${cleanKey}\n`;
    }
    fs.writeFileSync(dataEnv, currentContent, "utf-8");
  } catch (_) {}

  savePersistedData();
}

function setActiveOpenAiApiKey(key: string) {
  const cleanKey = (key || "").trim();
  if (!cleanKey) return;
  process.env.OPENAI_API_KEY = cleanKey;
  (appSettings as any).openai_api_key = cleanKey;

  let ep = modelEndpoints.find((e) => e.id === "openai-cloud" || e.provider === "openai");
  if (!ep) {
    ep = {
      id: "openai-cloud",
      name: "OpenAI",
      base_url: "https://api.openai.com/v1",
      api_key: cleanKey,
      api_key_fingerprint: cleanKey.slice(0, 4) + "...",
      has_key: true,
      provider: "openai",
      is_enabled: true,
      online: true,
      category: "api",
      model_type: "llm",
      models: ["gpt-4o", "gpt-4o-mini", "o3-mini"],
      models_display: ["GPT-4o", "GPT-4o Mini", "o3 Mini"],
      model_count: 3,
    };
    modelEndpoints.push(ep);
  } else {
    ep.api_key = cleanKey;
    ep.api_key_fingerprint = cleanKey.slice(0, 4) + "...";
    ep.has_key = true;
    ep.online = true;
  }

  try {
    const dataEnv = path.join(DATA_DIR, ".env");
    let currentContent = fs.existsSync(dataEnv) ? fs.readFileSync(dataEnv, "utf-8") : "";
    if (/^OPENAI_API_KEY=/m.test(currentContent)) {
      currentContent = currentContent.replace(/^OPENAI_API_KEY=.*$/m, `OPENAI_API_KEY=${cleanKey}`);
    } else {
      currentContent = (currentContent ? currentContent.trimEnd() + "\n" : "") + `OPENAI_API_KEY=${cleanKey}\n`;
    }
    fs.writeFileSync(dataEnv, currentContent, "utf-8");
  } catch (_) {}

  savePersistedData();
}

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
      if (Array.isArray(data.modelEndpoints) && data.modelEndpoints.length > 0) {
        modelEndpoints.length = 0;
        modelEndpoints.push(...data.modelEndpoints);
      }
      if (data.settings && typeof data.settings === "object") {
        appSettings = { ...appSettings, ...data.settings };
        if (data.settings.gemini_api_key && !process.env.GEMINI_API_KEY) {
          process.env.GEMINI_API_KEY = data.settings.gemini_api_key;
        }
      }
      // Inspect modelEndpoints for any stored Gemini key
      const storedGemini = modelEndpoints.find(
        (e) => (e.id === "gemini-cloud" || e.provider === "gemini") && e.api_key
      );
      if (storedGemini?.api_key && !process.env.GEMINI_API_KEY) {
        process.env.GEMINI_API_KEY = storedGemini.api_key;
      }
      // Ensure gemini-cloud reflects active key
      const activeGeminiKey = getActiveGeminiApiKey();
      const geminiEp = modelEndpoints.find((e) => e.id === "gemini-cloud" || e.provider === "gemini");
      if (geminiEp) {
        geminiEp.has_key = Boolean(activeGeminiKey);
        geminiEp.online = true;
        if (activeGeminiKey) {
          geminiEp.api_key = activeGeminiKey;
          geminiEp.api_key_fingerprint = activeGeminiKey.slice(0, 4) + "..." + activeGeminiKey.slice(-4);
        }
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
        modelEndpoints,
        settings: {
          ...appSettings,
          gemini_api_key: getActiveGeminiApiKey() || undefined,
        },
      };
      fs.writeFileSync(DATA_FILE, JSON.stringify(data, null, 2), "utf-8");
    } catch (err) {
      console.warn("[Storage] Failed to save persisted data:", err);
    }
  }, 250);
}

function initStorageDirectories() {
  try {
    const uploadsDir = path.join(DATA_DIR, "uploads");
    const docsDir = path.join(DATA_DIR, "documents");
    if (!fs.existsSync(DATA_DIR)) {
      fs.mkdirSync(DATA_DIR, { recursive: true });
    }
    if (!fs.existsSync(uploadsDir)) {
      fs.mkdirSync(uploadsDir, { recursive: true });
    }
    if (!fs.existsSync(docsDir)) {
      fs.mkdirSync(docsDir, { recursive: true });
    }
    if (!fs.existsSync(DATA_FILE)) {
      const data = {
        sessions: Array.from(sessions.values()),
        notes,
        tasks,
        calendarEvents,
        documents,
        memoryItems,
        modelEndpoints,
        settings: appSettings,
      };
      fs.writeFileSync(DATA_FILE, JSON.stringify(data, null, 2), "utf-8");
      console.log(`[Storage] Initialized persistent storage files and folders at ${DATA_DIR}`);
    }
  } catch (err) {
    console.warn("[Storage] Notice on initStorageDirectories:", err);
  }
}

// Initial load & directory initialization
loadPersistedData();
initStorageDirectories();

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
  const body = req.body || {};
  appSettings = { ...appSettings, ...body };

  if (body.gemini_api_key || body.GEMINI_API_KEY) {
    setActiveGeminiApiKey(body.gemini_api_key || body.GEMINI_API_KEY);
  }
  if (body.groq_api_key || body.GROQ_API_KEY) {
    setActiveGroqApiKey(body.groq_api_key || body.GROQ_API_KEY);
  }
  if (body.openai_api_key || body.OPENAI_API_KEY) {
    setActiveOpenAiApiKey(body.openai_api_key || body.OPENAI_API_KEY);
  }

  savePersistedData();
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
app.get("/api/models", (req, res) => {
  const items = modelEndpoints
    .filter((ep) => ep.is_enabled)
    .map((ep) => ({
      endpoint_id: ep.id,
      endpoint_name: ep.name,
      category: ep.category || "api",
      url: "/api/chat",
      model_type: ep.model_type || "llm",
      models: ep.models || [],
      models_display: ep.models_display || ep.models || [],
      models_extra: ep.models_extra || [],
      models_extra_display: ep.models_extra_display || [],
      offline: !ep.online,
    }));
  res.json({ items });
});

app.get("/api/model-endpoints", (req, res) => {
  res.json(modelEndpoints);
});

app.post("/api/model-endpoints/test", upload.any(), async (req, res) => {
  const body = req.body || {};
  let baseUrl = (body.base_url || "").toString().trim().replace(/\/+$/, "");
  const apiKey = (body.api_key || "").toString().trim();
  const provider = (body.provider || "").toString().trim().toLowerCase();

  const isGemini =
    provider === "gemini" ||
    provider === "google" ||
    baseUrl.includes("generativelanguage.googleapis.com") ||
    apiKey.startsWith("AIza");

  if (isGemini) {
    if (!apiKey) {
      return res.status(400).json({ ok: false, online: false, detail: "API key is required for Google Gemini", ping_error: "API key required" });
    }
    try {
      const testClient = new GoogleGenAI({ apiKey });
      const testResult = await testClient.models.generateContent({
        model: "gemini-2.5-flash",
        contents: [{ role: "user", parts: [{ text: "ping" }] }],
      });
      if (testResult) {
        setActiveGeminiApiKey(apiKey);
        return res.json({
          ok: true,
          online: true,
          status: "ok",
          models: ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-flash"],
        });
      }
    } catch (err: any) {
      console.warn("Gemini direct test error:", err?.message);
      return res.json({
        ok: false,
        online: false,
        detail: `Gemini verification failed: ${err?.message || "Invalid API key or network error"}`,
        ping_error: err?.message || "Test failed",
      });
    }
  }

  if (!baseUrl && provider === "groq") {
    baseUrl = "https://api.groq.com/openai/v1";
  }

  if (!baseUrl) {
    return res.status(400).json({ ok: false, online: false, detail: "Base URL is required", ping_error: "Base URL is required" });
  }

  try {
    const modelsUrl = baseUrl.endsWith("/v1")
      ? `${baseUrl}/models`
      : baseUrl.includes("/models")
      ? baseUrl
      : `${baseUrl}/v1/models`;

    const headers: Record<string, string> = { Accept: "application/json" };
    if (apiKey) {
      headers["Authorization"] = `Bearer ${apiKey}`;
    }

    const testRes = await fetch(modelsUrl, {
      method: "GET",
      headers,
      signal: AbortSignal.timeout(8000),
    });

    if (!testRes.ok) {
      const errText = await testRes.text().catch(() => "");
      return res.json({
        ok: false,
        online: false,
        detail: `HTTP ${testRes.status}: ${errText.slice(0, 120) || testRes.statusText}`,
        ping_error: `HTTP ${testRes.status}`,
      });
    }

    const data: any = await testRes.json();
    let modelList: string[] = [];
    if (Array.isArray(data.data)) {
      modelList = data.data.map((m: any) => m.id || m.name).filter(Boolean);
    } else if (Array.isArray(data.models)) {
      modelList = data.models.map((m: any) => m.name || m.id || m).filter(Boolean);
    }

    if (apiKey.startsWith("gsk_") || baseUrl.includes("groq.com")) {
      setActiveGroqApiKey(apiKey);
    } else if (apiKey.startsWith("sk-") || baseUrl.includes("openai.com")) {
      setActiveOpenAiApiKey(apiKey);
    }

    res.json({
      ok: true,
      online: true,
      status: modelList.length > 0 ? "ok" : "empty",
      models: modelList,
    });
  } catch (err: any) {
    res.json({
      ok: false,
      online: false,
      detail: err.name === "TimeoutError" ? "Connection timed out" : (err?.message || "Connection failed"),
      ping_error: err?.message || "Connection failed",
    });
  }
});

app.post("/api/model-endpoints", upload.any(), async (req, res) => {
  const body = req.body || {};
  let baseUrl = (body.base_url || "").toString().trim().replace(/\/+$/, "");
  const apiKey = (body.api_key || "").toString().trim();
  const provider = (body.provider || "").toString().trim().toLowerCase();
  const modelType = (body.model_type || "llm").toString();

  const isGemini =
    provider === "gemini" ||
    provider === "google" ||
    baseUrl.includes("generativelanguage.googleapis.com") ||
    apiKey.startsWith("AIza") ||
    (body.name && body.name.toString().toLowerCase().includes("gemini"));

  if (isGemini) {
    if (apiKey) {
      setActiveGeminiApiKey(apiKey);
    }
    const geminiEp = modelEndpoints.find((e) => e.id === "gemini-cloud" || e.provider === "gemini" || e.provider === "google");
    if (geminiEp) {
      return res.json({
        ...geminiEp,
        status: "ok",
        models: geminiEp.models || ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-flash"],
      });
    }
  }

  if (!baseUrl && provider === "groq") {
    baseUrl = "https://api.groq.com/openai/v1";
  }

  if (!baseUrl) {
    return res.status(400).json({ detail: "Base URL is required" });
  }

  let name = (body.name || "").toString().trim();
  if (!name) {
    if (baseUrl.includes("groq.com")) name = "Groq";
    else if (baseUrl.includes("openai.com")) name = "OpenAI";
    else if (baseUrl.includes("openrouter.ai")) name = "OpenRouter";
    else if (baseUrl.includes("anthropic.com")) name = "Anthropic";
    else if (baseUrl.includes("11434") || baseUrl.includes("ollama")) name = "Ollama";
    else {
      try {
        name = new URL(baseUrl).hostname;
      } catch (_) {
        name = "Custom Endpoint";
      }
    }
  }

  let models: string[] = [];
  try {
    const modelsUrl = baseUrl.endsWith("/v1")
      ? `${baseUrl}/models`
      : baseUrl.includes("/models")
      ? baseUrl
      : `${baseUrl}/v1/models`;
    const headers: Record<string, string> = { Accept: "application/json" };
    if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;

    const testRes = await fetch(modelsUrl, {
      method: "GET",
      headers,
      signal: AbortSignal.timeout(6000),
    });
    if (testRes.ok) {
      const data: any = await testRes.json();
      if (Array.isArray(data.data)) {
        models = data.data.map((m: any) => m.id || m.name).filter(Boolean);
      } else if (Array.isArray(data.models)) {
        models = data.models.map((m: any) => m.name || m.id || m).filter(Boolean);
      }
    }
  } catch (_) {}

  const endpoint: ModelEndpoint = {
    id: `ep-${Date.now()}`,
    name,
    base_url: baseUrl,
    api_key: apiKey || undefined,
    api_key_fingerprint: apiKey ? apiKey.slice(0, 4) + "..." + apiKey.slice(-4) : undefined,
    has_key: Boolean(apiKey),
    provider: provider || (baseUrl.includes("groq.com") ? "groq" : undefined),
    is_enabled: true,
    online: true,
    category: baseUrl.includes("localhost") || baseUrl.includes("127.0.0.1") || baseUrl.includes("host.docker.internal") ? "local" : "api",
    model_type: modelType === "image" ? "image" : "llm",
    models: models.length > 0 ? models : ["default"],
    models_display: models.length > 0 ? models : [name],
    model_count: models.length > 0 ? models.length : 1,
  };

  if (apiKey.startsWith("gsk_") || baseUrl.includes("groq.com")) {
    setActiveGroqApiKey(apiKey);
  } else if (apiKey.startsWith("sk-") || baseUrl.includes("openai.com")) {
    setActiveOpenAiApiKey(apiKey);
  }

  modelEndpoints.push(endpoint);
  savePersistedData();

  res.json({
    ...endpoint,
    status: models.length > 0 ? "ok" : "empty",
  });
});

app.patch("/api/model-endpoints/:id", upload.any(), (req, res) => {
  const ep = modelEndpoints.find((e) => e.id === req.params.id);
  if (!ep) {
    return res.status(404).json({ error: "Endpoint not found" });
  }
  ep.is_enabled = !ep.is_enabled;
  ep.online = true;
  savePersistedData();
  res.json(ep);
});

app.delete("/api/model-endpoints/:id", (req, res) => {
  const index = modelEndpoints.findIndex((ep) => ep.id === req.params.id);
  if (index !== -1) {
    modelEndpoints.splice(index, 1);
    savePersistedData();
  }
  res.json({ ok: true, id: req.params.id });
});

app.get("/api/model-endpoints/:id/models", (req, res) => {
  const ep = modelEndpoints.find((e) => e.id === req.params.id);
  if (!ep) return res.status(404).json({ error: "Endpoint not found" });
  res.json({ models: ep.models || [] });
});

app.get("/api/model-endpoints/:id/dependents", (req, res) => {
  res.json({ sessions: 0, models: 0 });
});

app.get("/api/model-endpoints/probe-local", (req, res) => {
  const result: Record<string, { alive: boolean }> = {};
  for (const ep of modelEndpoints) {
    result[ep.id] = { alive: true };
  }
  res.json(result);
});

app.post(["/api/probe-selected", "/api/probe"], (req, res) => {
  res.json({ ok: true, results: {} });
});

app.get("/api/providers", (req, res) => {
  res.json([
    { id: "groq", name: "Groq", base_url: "https://api.groq.com/openai/v1" },
    { id: "openai", name: "OpenAI", base_url: "https://api.openai.com/v1" },
    { id: "openrouter", name: "OpenRouter", base_url: "https://openrouter.ai/api/v1" },
    { id: "ollama", name: "Ollama", base_url: "http://localhost:11434/v1" },
    { id: "anthropic", name: "Anthropic", base_url: "https://api.anthropic.com/v1" },
  ]);
});

app.get("/api/default-chat", (req, res) => {
  const firstEnabled = modelEndpoints.find((e) => e.is_enabled);
  const model = firstEnabled?.models?.[0] || "gemini-2.5-flash";
  res.json({
    model,
    endpoint_id: firstEnabled?.id || "gemini-cloud",
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
app.post(["/api/chat_stream", "/api/chat"], upload.any(), async (req, res) => {
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

  let accumulatedReply = "";

  // Check if message itself is or contains an API key
  const geminiKeyMatch = messageText.match(/AIza[0-9A-Za-z_-]{35}/);
  if (geminiKeyMatch) {
    const key = geminiKeyMatch[0];
    setActiveGeminiApiKey(key);
    const reply =
      `✅ **Google Gemini API Key detected and activated successfully!**\n\n` +
      `Your key is saved to persistent storage. Gemini 2.5 Flash, Gemini 2.5 Pro, and Gemini 2.0 Flash are now ready.\n\n` +
      `You can now chat directly with Odysseus!`;
    const words = reply.split(" ");
    for (const w of words) {
      accumulatedReply += w + " ";
      res.write(`data: ${JSON.stringify({ delta: w + " " })}\n\n`);
      await new Promise((r) => setTimeout(r, 15));
    }
    session.messages.push({
      role: "assistant",
      content: accumulatedReply,
      timestamp: new Date().toISOString(),
    });
    session.last_message_at = new Date().toISOString();
    session.message_count = session.messages.length;
    res.write("data: [DONE]\n\n");
    res.end();
    return;
  }

  const groqKeyMatch = messageText.match(/gsk_[0-9A-Za-z_-]{20,}/);
  if (groqKeyMatch) {
    const key = groqKeyMatch[0];
    setActiveGroqApiKey(key);
    const reply = `✅ **Groq API Key detected and activated successfully!**\n\nYour Groq models (Llama 3.3 70B, Llama 3.1 8B, Mixtral 8x7B) are now online.`;
    const words = reply.split(" ");
    for (const w of words) {
      accumulatedReply += w + " ";
      res.write(`data: ${JSON.stringify({ delta: w + " " })}\n\n`);
      await new Promise((r) => setTimeout(r, 15));
    }
    session.messages.push({
      role: "assistant",
      content: accumulatedReply,
      timestamp: new Date().toISOString(),
    });
    session.last_message_at = new Date().toISOString();
    session.message_count = session.messages.length;
    res.write("data: [DONE]\n\n");
    res.end();
    return;
  }

  const geminiKey = getActiveGeminiApiKey();
  const ai = getGenAI();

  // 1. Check if model belongs to an OpenAI-compatible endpoint (Groq, OpenAI, Ollama, etc.)
  const targetEndpoint =
    modelEndpoints.find((ep) => ep.is_enabled && ep.base_url && ep.models?.includes(requestedModel)) ||
    (!requestedModel.startsWith("gemini")
      ? modelEndpoints.find((ep) => ep.is_enabled && ep.base_url && ep.id !== "gemini-cloud")
      : null);

  if (targetEndpoint && targetEndpoint.base_url && !requestedModel.startsWith("gemini")) {
    const completionsUrl = targetEndpoint.base_url.endsWith("/v1")
      ? `${targetEndpoint.base_url}/chat/completions`
      : targetEndpoint.base_url.includes("/chat/completions")
      ? targetEndpoint.base_url
      : `${targetEndpoint.base_url}/v1/chat/completions`;

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    };
    if (targetEndpoint.api_key) {
      headers["Authorization"] = `Bearer ${targetEndpoint.api_key}`;
    }

    const openAiMessages = session.messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    try {
      const apiRes = await fetch(completionsUrl, {
        method: "POST",
        headers,
        body: JSON.stringify({
          model: requestedModel,
          messages: openAiMessages,
          stream: true,
        }),
      });

      if (!apiRes.ok) {
        const errBody = await apiRes.text().catch(() => "");
        throw new Error(`HTTP ${apiRes.status}: ${errBody.slice(0, 150)}`);
      }

      const reader = apiRes.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith("data:")) continue;
          const raw = trimmed.slice(5).trim();
          if (raw === "[DONE]") continue;
          try {
            const parsed = JSON.parse(raw);
            const delta = parsed.choices?.[0]?.delta?.content || "";
            if (delta) {
              accumulatedReply += delta;
              res.write(`data: ${JSON.stringify({ delta })}\n\n`);
            }
          } catch (_) {}
        }
      }
    } catch (err: any) {
      console.error(`${targetEndpoint.name} streaming error:`, err);
      const errMessage = `\n\n*(${targetEndpoint.name} error: ${err?.message || "Failed to generate response"})*`;
      accumulatedReply += errMessage;
      res.write(`data: ${JSON.stringify({ delta: errMessage })}\n\n`);
    }
  } else if (ai && geminiKey) {
    try {
      // Prepare conversation history for Gemini
      const contents = session.messages
        .filter((m) => m.content && m.content.trim() && (m.role === "user" || m.role === "assistant"))
        .map((m) => ({
          role: m.role === "assistant" ? "model" : "user",
          parts: [{ text: m.content }],
        }));

      // Ensure the current user message is at the end of the history
      const lastMsg = contents[contents.length - 1];
      if (!lastMsg || lastMsg.role !== "user" || lastMsg.parts[0]?.text !== messageText) {
        contents.push({
          role: "user",
          parts: [{ text: messageText }],
        });
      }

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
      const errMessage = `\n\n*(Gemini API error: ${err?.message || "Failed to generate response"}. Please check your Gemini API key.)*`;
      accumulatedReply += errMessage;
      res.write(`data: ${JSON.stringify({ delta: errMessage })}\n\n`);
    }
  } else {
    // Graceful fallback when GEMINI_API_KEY is not yet populated
    const fallbackText =
      `Hello! I received your message: "${messageText}".\n\n` +
      `Odysseus is running successfully, but your **Gemini API Key** is not yet active.\n\n` +
      `### ⚡ Quick Ways to Activate Gemini:\n` +
      `1. **In this Chat**: Simply paste your key (starting with \`AIza...\`) directly into this chat box, or run: \`/setup gemini <YOUR_KEY>\`\n` +
      `2. **In Settings**: Click the **⚙️ Settings** icon (in the bottom left) → **Services** → **Add API Models** → Pick **Google Gemini**, paste your key, and click **Add**.\n` +
      `3. **In Docker**: Add \`GEMINI_API_KEY=YOUR_KEY\` to your stack environment variables or put it in \`./data/.env\`.`;

    const words = fallbackText.split(" ");
    for (const word of words) {
      const delta = word + " ";
      accumulatedReply += delta;
      res.write(`data: ${JSON.stringify({ delta })}\n\n`);
      await new Promise((r) => setTimeout(r, 20));
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
  const search = typeof req.query.search === "string" ? req.query.search.trim().toLowerCase() : "";
  const language = typeof req.query.language === "string" ? req.query.language.trim() : "";
  const sort = typeof req.query.sort === "string" ? req.query.sort : "recent";
  const offset = parseInt(String(req.query.offset || "0"), 10) || 0;
  const limit = Math.min(Math.max(parseInt(String(req.query.limit || "50"), 10) || 50, 1), 100);
  const archived = req.query.archived === "true";

  // Filter archived vs active
  const baseList = documents.filter((d) => (archived ? !!d.archived : !d.archived));

  // Compute language facets from the active/archived base set
  const languages: Record<string, number> = {};
  const sessionIds = new Set<string>();

  for (const doc of baseList) {
    const lang = doc.language || "text";
    languages[lang] = (languages[lang] || 0) + 1;
    if (doc.session_id) {
      sessionIds.add(doc.session_id);
    }
  }

  // Filter by search and language
  let filtered = [...baseList];
  if (search) {
    const tokens = search.split(/\s+/).filter(Boolean);
    filtered = filtered.filter((d) => {
      const titleLower = (d.title || "").toLowerCase();
      const contentLower = (d.content || d.current_content || "").toLowerCase();
      return tokens.every((tok) => titleLower.includes(tok) || contentLower.includes(tok));
    });
  }

  if (language) {
    if (language === "text") {
      filtered = filtered.filter((d) => !d.language || d.language === "text");
    } else {
      filtered = filtered.filter((d) => d.language === language);
    }
  }

  // Sorting
  if (sort === "oldest") {
    filtered.sort((a, b) => (a.created_at || a.updated_at).localeCompare(b.created_at || b.updated_at));
  } else if (sort === "alpha") {
    filtered.sort((a, b) => (a.title || "").localeCompare(b.title || ""));
  } else if (sort === "edits") {
    filtered.sort((a, b) => (b.version_count || 1) - (a.version_count || 1));
  } else {
    // recent
    filtered.sort((a, b) => (b.updated_at || "").localeCompare(a.updated_at || ""));
  }

  const total = filtered.length;
  const paged = filtered.slice(offset, offset + limit).map((d) => ({
    id: d.id,
    session_id: d.session_id,
    session_name: d.session_name || (d.session_id ? sessions.get(d.session_id)?.name : undefined),
    title: d.title,
    language: d.language || "text",
    preview: d.preview || (d.content || d.current_content || "").slice(0, 500),
    version_count: d.version_count || 1,
    created_at: d.created_at || d.updated_at,
    updated_at: d.updated_at,
    content: d.content || d.current_content || "",
    current_content: d.current_content || d.content || "",
    archived: !!d.archived,
  }));

  res.json({
    documents: paged,
    total,
    languages,
    session_count: sessionIds.size,
  });
});

app.get("/api/documents/:sessionId", (req, res) => {
  const list = documents
    .filter((d) => d.session_id === req.params.sessionId && !d.archived)
    .map((d) => ({
      ...d,
      current_content: d.current_content || d.content || "",
      content: d.content || d.current_content || "",
    }));
  res.json(list);
});

app.get("/api/document/:id", (req, res) => {
  const doc = documents.find((d) => d.id === req.params.id);
  if (!doc) {
    return res.status(404).json({ detail: "Document not found" });
  }
  res.json({
    ...doc,
    current_content: doc.current_content || doc.content || "",
    content: doc.content || doc.current_content || "",
  });
});

app.post("/api/document", (req, res) => {
  const { title = "Untitled", content = "", language = "text", session_id } = req.body || {};
  const now = new Date().toISOString();
  const session = session_id ? sessions.get(session_id) : undefined;
  const doc: DocumentItem = {
    id: crypto.randomUUID(),
    session_id: session_id || undefined,
    session_name: session ? session.name : undefined,
    title: title || "Untitled",
    content: content || "",
    current_content: content || "",
    language: language || "text",
    preview: (content || "").slice(0, 500),
    version_count: 1,
    is_active: true,
    archived: false,
    created_at: now,
    updated_at: now,
  };
  documents.unshift(doc);
  savePersistedData();
  res.json(doc);
});

app.put("/api/document/:id", (req, res) => {
  const doc = documents.find((d) => d.id === req.params.id);
  if (!doc) {
    return res.status(404).json({ detail: "Document not found" });
  }
  const { title, content, language } = req.body || {};
  if (title !== undefined) doc.title = title;
  if (content !== undefined) {
    doc.content = content;
    doc.current_content = content;
    doc.preview = content.slice(0, 500);
    doc.version_count = (doc.version_count || 1) + 1;
  }
  if (language !== undefined) doc.language = language;
  doc.updated_at = new Date().toISOString();
  savePersistedData();
  res.json({
    ...doc,
    current_content: doc.content,
    content: doc.content,
  });
});

app.patch("/api/document/:id", (req, res) => {
  const doc = documents.find((d) => d.id === req.params.id);
  if (!doc) {
    return res.status(404).json({ detail: "Document not found" });
  }
  const { title, content, language } = req.body || {};
  if (title !== undefined) doc.title = title;
  if (content !== undefined) {
    doc.content = content;
    doc.current_content = content;
    doc.preview = content.slice(0, 500);
  }
  if (language !== undefined) doc.language = language;
  doc.updated_at = new Date().toISOString();
  savePersistedData();
  res.json({
    ...doc,
    current_content: doc.content,
    content: doc.content,
  });
});

app.delete("/api/document/:id", (req, res) => {
  const index = documents.findIndex((d) => d.id === req.params.id);
  if (index !== -1) {
    documents.splice(index, 1);
    savePersistedData();
  }
  res.json({ ok: true, id: req.params.id });
});

app.post("/api/document/:id/archive", (req, res) => {
  const doc = documents.find((d) => d.id === req.params.id);
  if (!doc) {
    return res.status(404).json({ detail: "Document not found" });
  }
  const toArchive = req.query.archived !== "false" && req.body?.archived !== false;
  doc.archived = toArchive;
  doc.updated_at = new Date().toISOString();
  savePersistedData();
  res.json({ ok: true, id: doc.id, archived: doc.archived });
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
  res.json({ images: [], items: [], total: 0 });
});

app.get("/api/research/library", (req, res) => {
  res.json({ items: [], research: [], total: 0 });
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
