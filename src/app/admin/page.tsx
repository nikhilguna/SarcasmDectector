import { createServiceClient } from '@/lib/supabase'

export const dynamic = 'force-dynamic'
export const revalidate = 0

interface Session {
  id: string
  worker_id: string
  condition: string
  completion_code: string
  trials_completed: number
  accuracy: number
  created_at: string
}

interface Response {
  condition: string
  participant_answer: string
  ground_truth: string
}

export default async function AdminPage() {
  const supabase = createServiceClient()

  // Fetch sessions
  const { data: sessions } = await supabase
    .from('sessions')
    .select('*')
    .order('created_at', { ascending: false })

  // Fetch response stats
  const { data: responses } = await supabase
    .from('responses')
    .select('condition, participant_answer, ground_truth')

  const sessionList = (sessions || []) as Session[]
  const responseList = (responses || []) as Response[]

  // Calculate stats
  const stats = {
    total: sessionList.length,
    noAI: {
      count: sessionList.filter(s => s.condition === 'no-ai').length,
      correct: responseList.filter(r => r.condition === 'no-ai' && r.participant_answer === r.ground_truth).length,
      total: responseList.filter(r => r.condition === 'no-ai').length
    },
    withAI: {
      count: sessionList.filter(s => s.condition === 'with-ai').length,
      correct: responseList.filter(r => r.condition === 'with-ai' && r.participant_answer === r.ground_truth).length,
      total: responseList.filter(r => r.condition === 'with-ai').length
    }
  }

  const noAIAccuracy = stats.noAI.total > 0
    ? ((stats.noAI.correct / stats.noAI.total) * 100).toFixed(1)
    : 'N/A'
  const withAIAccuracy = stats.withAI.total > 0
    ? ((stats.withAI.correct / stats.withAI.total) * 100).toFixed(1)
    : 'N/A'

  return (
    <main className="max-w-4xl mx-auto p-6">
      <h1 className="text-2xl font-semibold mb-2">Sarcasm Detection Study</h1>
      <p className="text-gray-500 mb-6">
        Admin Dashboard &middot;{' '}
        <a href="/admin" className="text-blue-600 hover:underline">Refresh</a>
      </p>

      {/* Summary Stats */}
      <div className="bg-white rounded-xl p-6 shadow-sm mb-6">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-3xl font-bold text-blue-600">{stats.total}</div>
            <div className="text-sm text-gray-500">Total Participants</div>
          </div>
          <div>
            <div className="text-3xl font-bold text-blue-600">{stats.noAI.count}</div>
            <div className="text-sm text-gray-500">No-AI Condition</div>
          </div>
          <div>
            <div className="text-3xl font-bold text-blue-600">{stats.withAI.count}</div>
            <div className="text-sm text-gray-500">With-AI Condition</div>
          </div>
        </div>
      </div>

      {/* Accuracy */}
      <h2 className="text-lg font-semibold text-gray-700 mb-3">Accuracy by Condition</h2>
      <div className="bg-white rounded-xl p-6 shadow-sm mb-6">
        <div className="grid grid-cols-2 gap-4 text-center">
          <div>
            <div className="text-3xl font-bold text-blue-600">{noAIAccuracy}%</div>
            <div className="text-sm text-gray-500">
              No-AI ({stats.noAI.correct}/{stats.noAI.total} trials)
            </div>
          </div>
          <div>
            <div className="text-3xl font-bold text-blue-600">{withAIAccuracy}%</div>
            <div className="text-sm text-gray-500">
              With-AI ({stats.withAI.correct}/{stats.withAI.total} trials)
            </div>
          </div>
        </div>
      </div>

      {/* Individual Participants */}
      <h2 className="text-lg font-semibold text-gray-700 mb-3">Individual Participants</h2>
      <div className="bg-white rounded-xl shadow-sm overflow-hidden">
        {sessionList.length === 0 ? (
          <p className="p-6 text-gray-500">No submissions yet.</p>
        ) : (
          <table className="w-full">
            <thead className="bg-gray-50 text-left text-sm text-gray-600">
              <tr>
                <th className="px-4 py-3 font-medium">Participant ID</th>
                <th className="px-4 py-3 font-medium">Condition</th>
                <th className="px-4 py-3 font-medium">Accuracy</th>
                <th className="px-4 py-3 font-medium">Code</th>
                <th className="px-4 py-3 font-medium">Timestamp</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {sessionList.map((s) => (
                <tr key={s.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-sm">{s.worker_id}</td>
                  <td className="px-4 py-3">
                    <span className={`text-xs font-medium px-2 py-1 rounded ${
                      s.condition === 'with-ai'
                        ? 'bg-blue-100 text-blue-700'
                        : 'bg-gray-100 text-gray-700'
                    }`}>
                      {s.condition}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm">
                    {((s.accuracy || 0) * 100).toFixed(1)}%
                  </td>
                  <td className="px-4 py-3 text-sm font-mono">{s.completion_code}</td>
                  <td className="px-4 py-3 text-sm text-gray-500">
                    {new Date(s.created_at).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </main>
  )
}
