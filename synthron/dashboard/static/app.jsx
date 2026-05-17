/* ─── Synthron Live Dashboard ─────────────────────────────── */
const { useState, useEffect, useRef, useCallback } = React;

const AGENT_COLORS = {
  planner: '#06b6d4', executor: '#10b981', critic: '#f59e0b',
  memory: '#8b5cf6', researcher: '#3b82f6', coder: '#f1f5f9',
  coordinator: '#ef4444', orchestrator: '#6366f1',
};

const EVENT_ICONS = {
  thought: '💭', action: '🔧', result: '✅', score: '📊',
  error: '❌', plan_created: '📋', task_start: '▶', task_done: '🏁',
  retry: '🔄', executing: '⚡', coding: '💻', researching: '🔍',
  memory_store: '💾', memory_recall: '🧠', coordinating: '🎯',
};

/* ─── WebSocket hook ─────────────────────────────────────── */
function useWebSocket(url) {
  const [events, setEvents] = useState([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    ws.onmessage = (e) => {
      try {
        const event = JSON.parse(e.data);
        if (event.type === 'keepalive') return;
        setEvents(prev => [...prev.slice(-200), { ...event, id: Date.now() }]);
      } catch {}
    };

    const ping = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) ws.send('ping');
    }, 25000);

    return () => { clearInterval(ping); ws.close(); };
  }, [url]);

  return { events, connected };
}

/* ─── AgentGraph Component ───────────────────────────────── */
function AgentGraph({ events }) {
  const agentTypes = ['orchestrator', 'planner', 'executor', 'critic', 'memory', 'researcher', 'coder'];

  const getAgentStatus = (agentType) => {
    const recent = events.slice(-30).filter(e => e.agent_type === agentType || e.agent === agentType);
    if (recent.length === 0) return 'idle';
    const last = recent[recent.length - 1];
    if (last.type === 'error') return 'failed';
    if (['task_done', 'subtask_done', 'result'].includes(last.type)) return 'idle';
    return 'running';
  };

  return (
    <div className="card">
      <h3>Agent Graph</h3>
      {agentTypes.map(agent => {
        const status = getAgentStatus(agent);
        const color = AGENT_COLORS[agent] || '#94a3b8';
        return (
          <div key={agent} className={`agent-node ${status}`}>
            <div className="dot" style={status === 'idle' ? { background: color, opacity: 0.4 } : { background: color }} />
            <span style={{ color: status === 'idle' ? '#94a3b8' : color, fontWeight: status === 'running' ? 700 : 400 }}>
              {agent.charAt(0).toUpperCase() + agent.slice(1)}
            </span>
            <span style={{ marginLeft: 'auto', fontSize: 10, color: '#64748b' }}>
              {status}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/* ─── ThoughtStream Component ────────────────────────────── */
function ThoughtStream({ events }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [events]);

  return (
    <div className="thought-stream">
      {events.length === 0 && (
        <div style={{ color: '#475569', textAlign: 'center', marginTop: 40, fontFamily: 'monospace' }}>
          <div style={{ fontSize: 32, marginBottom: 12 }}>🧠</div>
          <div>Waiting for agent activity...</div>
          <div style={{ fontSize: 12, marginTop: 8 }}>Submit a task below to start</div>
        </div>
      )}
      {events.map(event => {
        const icon = EVENT_ICONS[event.type] || '·';
        const agentColor = AGENT_COLORS[event.agent_type] || AGENT_COLORS[event.agent] || '#94a3b8';
        const tagClass = `tag-${event.agent_type || event.agent || 'orchestrator'}`;
        const entryClass = `thought-entry ${event.type || 'thought'}`;

        return (
          <div key={event.id} className={entryClass}>
            <span className={`agent-tag ${tagClass}`}>
              [{(event.agent || 'system').toUpperCase()}]
            </span>
            <span style={{ marginRight: 6 }}>{icon}</span>
            <span>{event.content}</span>
          </div>
        );
      })}
      <div ref={bottomRef} />
    </div>
  );
}

/* ─── MetricsPanel Component ─────────────────────────────── */
function MetricsPanel({ events, connected }) {
  const successCount = events.filter(e => e.type === 'task_done').length;
  const errorCount = events.filter(e => e.type === 'task_error' || e.type === 'error').length;
  const retryCount = events.filter(e => e.type === 'retry').length;
  const scoreEvents = events.filter(e => e.type === 'score');

  const avgScore = scoreEvents.length > 0
    ? (scoreEvents.reduce((sum, e) => {
        const m = e.content.match(/(\d+\.\d+)/);
        return sum + (m ? parseFloat(m[1]) : 0);
      }, 0) / scoreEvents.length).toFixed(2)
    : '—';

  const providers = ['gemini', 'groq', 'cerebras', 'deepseek', 'openrouter', 'github', 'ollama'];

  return (
    <>
      <div className="card">
        <h3>Session Stats</h3>
        <div className="metric-grid">
          <div className="metric-item">
            <div className="value" style={{ color: '#10b981' }}>{successCount}</div>
            <div className="label">Tasks Done</div>
          </div>
          <div className="metric-item">
            <div className="value" style={{ color: '#ef4444' }}>{errorCount}</div>
            <div className="label">Errors</div>
          </div>
          <div className="metric-item">
            <div className="value" style={{ color: '#f59e0b' }}>{retryCount}</div>
            <div className="label">Retries</div>
          </div>
          <div className="metric-item">
            <div className="value" style={{ color: '#6366f1' }}>{avgScore}</div>
            <div className="label">Avg Score</div>
          </div>
        </div>
      </div>

      <div className="card">
        <h3>Providers</h3>
        <div className="provider-list">
          {providers.map(p => (
            <div key={p} className={`provider-item ${connected ? 'active' : ''}`}>
              <span style={{ textTransform: 'capitalize' }}>{p}</span>
              <span className="badge">{connected ? 'ready' : 'offline'}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <h3>Connection</h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{
            width: 10, height: 10, borderRadius: '50%',
            background: connected ? '#10b981' : '#ef4444',
          }} />
          <span style={{ fontSize: 13, color: connected ? '#10b981' : '#ef4444' }}>
            {connected ? 'WebSocket Connected' : 'Disconnected'}
          </span>
        </div>
        <div style={{ marginTop: 8, fontSize: 11, color: '#475569' }}>
          Events received: {events.length}
        </div>
      </div>
    </>
  );
}

/* ─── TaskInput Component ────────────────────────────────── */
function TaskInput({ onSubmit, isRunning }) {
  const [task, setTask] = useState('');

  const handleSubmit = () => {
    if (!task.trim() || isRunning) return;
    onSubmit(task.trim());
    setTask('');
  };

  const examples = [
    'Research the top 5 AI companies in 2026 and their funding',
    'Write a Python script to analyze stock price data',
    'Summarize the latest developments in quantum computing',
  ];

  return (
    <div className="task-input-area">
      <div className="input-row">
        <input
          type="text"
          value={task}
          onChange={e => setTask(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSubmit()}
          placeholder="Enter your task... (Press Enter to run)"
          disabled={isRunning}
        />
        <button className="btn btn-primary" onClick={handleSubmit} disabled={isRunning || !task.trim()}>
          {isRunning ? '⚡ Running...' : '▶ Run'}
        </button>
      </div>
      <div style={{ marginTop: 8, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {examples.map(ex => (
          <button
            key={ex}
            onClick={() => setTask(ex)}
            style={{
              background: 'none', border: '1px solid #2d3748',
              borderRadius: 4, padding: '3px 8px', fontSize: 11,
              color: '#94a3b8', cursor: 'pointer',
            }}
          >
            {ex.slice(0, 40)}...
          </button>
        ))}
      </div>
    </div>
  );
}

/* ─── App Root ───────────────────────────────────────────── */
function App() {
  const wsProto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const wsUrl = `${wsProto}://${window.location.host}/ws/events/all`;
  const { events, connected } = useWebSocket(wsUrl);
  const [isRunning, setIsRunning] = useState(false);
  const [localEvents, setLocalEvents] = useState([]);
  const allEvents = [...events, ...localEvents].sort((a, b) => (a.id || 0) - (b.id || 0));

  const handleTask = async (task) => {
    setIsRunning(true);
    setLocalEvents(prev => [...prev, {
      id: Date.now(),
      type: 'task_start',
      agent: 'dashboard',
      agent_type: 'orchestrator',
      content: `Submitting: ${task}`,
    }]);

    try {
      const resp = await fetch('/api/v1/tasks/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task }),
      });
      const result = await resp.json();
      setLocalEvents(prev => [...prev, {
        id: Date.now(),
        type: result.success ? 'task_done' : 'error',
        agent: 'orchestrator',
        agent_type: 'orchestrator',
        content: result.success
          ? `✅ Done in ${result.total_time_s}s | ${result.total_tokens.toLocaleString()} tokens`
          : `❌ ${result.error}`,
      }]);
    } catch (e) {
      setLocalEvents(prev => [...prev, {
        id: Date.now(),
        type: 'error',
        agent: 'dashboard',
        agent_type: 'orchestrator',
        content: `API error: ${e.message}`,
      }]);
    }
    setIsRunning(false);
  };

  return (
    <div className="dashboard">
      <header className="header">
        <div className="status-dot" />
        <h1>SYNTHRON</h1>
        <span style={{ color: '#475569', fontSize: 13 }}>The Neural Fabric for Autonomous AI Agents</span>
        <span style={{ marginLeft: 'auto', fontSize: 12, color: connected ? '#10b981' : '#ef4444' }}>
          {connected ? '● LIVE' : '○ OFFLINE'}
        </span>
      </header>

      <aside className="sidebar-left">
        <AgentGraph events={allEvents} />
      </aside>

      <main className="main-panel">
        <ThoughtStream events={allEvents} />
        <TaskInput onSubmit={handleTask} isRunning={isRunning} />
      </main>

      <aside className="sidebar-right">
        <MetricsPanel events={allEvents} connected={connected} />
      </aside>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
