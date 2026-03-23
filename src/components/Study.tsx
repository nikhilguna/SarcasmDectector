'use client'

import { useState, useEffect, useRef } from 'react'
import { supabase } from '@/lib/supabase'

interface Trial {
  trial_id: number
  context: string[]
  response: string
  ground_truth: string
  model_prediction?: string
  model_confidence?: number
  model_reasoning?: string
}

interface StudyProps {
  condition: 'no-ai' | 'with-ai'
}

type Phase = 'loading' | 'instructions' | 'trials' | 'complete'

interface Response {
  trial_id: number
  ground_truth: string
  participant_answer: string
  time_spent_ms: number
}

export default function Study({ condition }: StudyProps) {
  const [phase, setPhase] = useState<Phase>('loading')
  const [workerId, setWorkerId] = useState<string>('')
  const [trials, setTrials] = useState<Trial[]>([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [responses, setResponses] = useState<Response[]>([])
  const [completionCode, setCompletionCode] = useState('')
  const [copied, setCopied] = useState(false)
  const trialStartTime = useRef<number>(0)

  useEffect(() => {
    initSession()
  }, [condition])

  async function initSession() {
    try {
      const res = await fetch(`/api/session?condition=${condition}`)
      const data = await res.json()
      setWorkerId(data.worker_id)
      setTrials(data.trials)
      setPhase('instructions')
    } catch (err) {
      console.error('Failed to init session:', err)
    }
  }

  function startStudy() {
    setPhase('trials')
    trialStartTime.current = Date.now()
  }

  async function handleAnswer(answer: 'SARCASM' | 'NOT_SARCASM') {
    const trial = trials[currentIndex]
    const timeSpent = Date.now() - trialStartTime.current

    const response: Response = {
      trial_id: trial.trial_id,
      ground_truth: trial.ground_truth,
      participant_answer: answer,
      time_spent_ms: timeSpent
    }

    // Save to Supabase immediately
    await supabase.from('responses').insert({
      worker_id: workerId,
      condition,
      trial_id: trial.trial_id,
      ground_truth: trial.ground_truth,
      participant_answer: answer,
      time_spent_ms: timeSpent
    })

    const newResponses = [...responses, response]
    setResponses(newResponses)

    if (currentIndex + 1 < trials.length) {
      setCurrentIndex(currentIndex + 1)
      trialStartTime.current = Date.now()
    } else {
      // Study complete - submit session
      await completeStudy(newResponses)
    }
  }

  async function completeStudy(allResponses: Response[]) {
    try {
      const res = await fetch('/api/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          worker_id: workerId,
          condition,
          responses: allResponses
        })
      })
      const data = await res.json()
      setCompletionCode(data.completion_code)
      setPhase('complete')
    } catch (err) {
      console.error('Submit error:', err)
      setCompletionCode('ERROR')
      setPhase('complete')
    }
  }

  function copyCode() {
    navigator.clipboard.writeText(completionCode)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const showAI = condition === 'with-ai'
  const trial = trials[currentIndex]
  const progress = trials.length > 0 ? ((currentIndex + 1) / trials.length) * 100 : 0

  if (phase === 'loading') {
    return (
      <main className="max-w-2xl mx-auto p-6">
        <div className="bg-white rounded-xl p-12 shadow-sm text-center">
          <div className="w-10 h-10 border-4 border-gray-200 border-t-blue-600 rounded-full animate-spin mx-auto" />
          <p className="mt-4 text-gray-500">Loading study...</p>
        </div>
      </main>
    )
  }

  if (phase === 'instructions') {
    return (
      <main className="max-w-2xl mx-auto p-6">
        <h1 className="text-2xl font-semibold mb-6">Task Instructions</h1>

        <div className="bg-white rounded-xl p-6 shadow-sm">
          <h2 className="text-lg font-semibold mb-4">Detecting Sarcasm in Reddit Comments</h2>

          <p className="text-gray-600 mb-4">
            In this study, you will read <strong>20 Reddit conversation threads</strong> and
            classify the final response as either <strong>Sarcastic</strong> or <strong>Not Sarcastic</strong>.
          </p>

          <h3 className="font-medium text-gray-700 mt-6 mb-2">What is Sarcasm?</h3>
          <p className="text-gray-600 mb-2">
            Sarcasm is when someone says the <em>opposite</em> of what they mean, often to mock, criticize, or be humorous.
          </p>
          <ul className="list-disc list-inside text-gray-600 mb-4 space-y-1">
            <li>&quot;Oh great, another meeting. Just what I needed.&quot;</li>
            <li>&quot;Wow, you&apos;re a real genius.&quot; (when someone did something foolish)</li>
          </ul>

          <h3 className="font-medium text-gray-700 mt-6 mb-2">How Each Trial Works</h3>
          <ol className="list-decimal list-inside text-gray-600 space-y-1 mb-4">
            <li>You&apos;ll see the <strong>thread context</strong> (previous comments in gray)</li>
            <li>Below that, you&apos;ll see the <strong>target response</strong> to classify</li>
            {showAI && (
              <li>You&apos;ll also see an <strong>AI prediction</strong> with confidence and reasoning</li>
            )}
            <li>Click <strong>&quot;Sarcastic&quot;</strong> or <strong>&quot;Not Sarcastic&quot;</strong></li>
          </ol>

          <p className="text-sm text-gray-500 italic mb-6">
            This study takes approximately 10-15 minutes to complete.
          </p>

          <button
            onClick={startStudy}
            className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition"
          >
            Start the Study
          </button>
        </div>
      </main>
    )
  }

  if (phase === 'trials' && trial) {
    return (
      <main className="max-w-2xl mx-auto p-6">
        <div className="mb-4">
          <p className="text-sm text-gray-500 mb-2">Trial {currentIndex + 1} of {trials.length}</p>
          <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-600 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm overflow-hidden">
          {/* Context */}
          <div className="bg-gray-50 p-4 border-b">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-3">
              Thread Context
            </h3>
            <div className="space-y-2">
              {trial.context.map((comment, i) => (
                <div key={i} className="bg-gray-200 rounded-lg p-3">
                  <span className="text-xs font-medium text-gray-500 block mb-1">
                    Comment {i + 1}
                  </span>
                  <p className="text-gray-700">{comment}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Target Response */}
          <div className="p-4 border-b">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-emerald-600 mb-3">
              Response to Classify
            </h3>
            <div className="bg-emerald-50 border-2 border-emerald-500 rounded-lg p-4">
              <p className="text-emerald-800">{trial.response}</p>
            </div>
          </div>

          {/* AI Section */}
          {showAI && trial.model_prediction && (
            <div className="bg-blue-50 p-4 border-b border-blue-100">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-blue-600 mb-3">
                AI Assistant
              </h3>
              <div className="bg-white border border-blue-200 rounded-lg p-4 space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-gray-600">Prediction:</span>
                  <span className={`font-semibold ${
                    trial.model_prediction === 'SARCASM' ? 'text-red-600' : 'text-emerald-600'
                  }`}>
                    {trial.model_prediction === 'SARCASM' ? 'Sarcastic' : 'Not Sarcastic'}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-gray-600">Confidence:</span>
                  <span className="font-semibold text-blue-600">{trial.model_confidence}%</span>
                </div>
                <div className="pt-2 border-t">
                  <span className="text-sm font-medium text-gray-600 block mb-1">Reasoning:</span>
                  <p className="text-sm text-gray-600">{trial.model_reasoning}</p>
                </div>
              </div>
            </div>
          )}

          {/* Answer Buttons */}
          <div className="p-4 bg-gray-50">
            <p className="text-center font-medium text-gray-700 mb-3">
              Is this response sarcastic?
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => handleAnswer('SARCASM')}
                className="flex-1 py-3 bg-white border-2 border-gray-200 hover:border-blue-500 rounded-lg font-medium transition"
              >
                Sarcastic
              </button>
              <button
                onClick={() => handleAnswer('NOT_SARCASM')}
                className="flex-1 py-3 bg-white border-2 border-gray-200 hover:border-blue-500 rounded-lg font-medium transition"
              >
                Not Sarcastic
              </button>
            </div>
          </div>
        </div>
      </main>
    )
  }

  if (phase === 'complete') {
    return (
      <main className="max-w-2xl mx-auto p-6">
        <h1 className="text-2xl font-semibold mb-6">Study Complete!</h1>

        <div className="bg-white rounded-xl p-8 shadow-sm text-center">
          <div className="w-16 h-16 bg-emerald-500 rounded-full flex items-center justify-center mx-auto mb-4">
            <span className="text-white text-3xl">&#10003;</span>
          </div>

          <h2 className="text-xl font-semibold mb-4">Thank you for participating!</h2>

          <p className="text-gray-600 mb-2">Your completion code is:</p>
          <div className="bg-gray-100 border-2 border-dashed border-gray-300 rounded-lg p-4 mb-4">
            <code className="text-2xl font-bold tracking-wider">{completionCode}</code>
          </div>

          <p className="text-sm text-gray-500 mb-4">
            <strong>Important:</strong> Copy this code and paste it back into MTurk to receive credit.
          </p>

          <button
            onClick={copyCode}
            className="px-6 py-2 bg-gray-200 hover:bg-gray-300 rounded-lg font-medium transition"
          >
            {copied ? 'Copied!' : 'Copy Code'}
          </button>
        </div>
      </main>
    )
  }

  return null
}
