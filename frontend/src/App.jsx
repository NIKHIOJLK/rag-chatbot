import { useState } from 'react'
import './App.css'

// Backend address: set VITE_API_URL in frontend/.env for deployment
const API_URL = (import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000').replace(/\/$/, '')

function App() {
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)

  const askQuestion = async () => {
    if (!question.trim()) return

    const userMessage = { role: 'user', text: question }
    setMessages(prev => [...prev, userMessage])
    setLoading(true)
    setQuestion('')

    try {
      // send the last few turns so follow-up questions make sense
      const history = messages.slice(-6).map(m => ({ role: m.role === 'user' ? 'user' : 'assistant', content: m.text }))
      const response = await fetch(`${API_URL}/api/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMessage.text, history })
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Request failed')

      const botMessage = {
        role: 'bot',
        text: data.answer,
        sources: data.sources
      }
      setMessages(prev => [...prev, botMessage])
    } catch (err) {
      const text = err instanceof TypeError ? 'Error: could not reach the server. Is the backend running?' : `Error: ${err.message}`
      setMessages(prev => [...prev, { role: 'bot', text }])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') askQuestion()
  }

  return (
    <div className="chat-container">
      <h1>Python Docs Q&A</h1>
      <p className="subtitle">Ask a question about Python — answered using retrieved documentation.</p>

      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-text">{msg.text}</div>
            {msg.sources && msg.sources.length > 0 && (
              <div className="sources">
                Sources:
                {msg.sources.map((src) => (
                  <a key={src.n} href={src.url} target="_blank" rel="noreferrer">[{src.n}] {src.title}</a>
                ))}
              </div>
            )}
          </div>
        ))}
        {loading && <div className="message bot">Thinking...</div>}
      </div>

      <div className="input-row">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="e.g. What is a list comprehension?"
          disabled={loading}
        />
        <button onClick={askQuestion} disabled={loading}>Ask</button>
      </div>
    </div>
  )
}

export default App