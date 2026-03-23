import Link from 'next/link'

export default function Home() {
  return (
    <main className="max-w-2xl mx-auto p-6">
      <h1 className="text-2xl font-semibold mb-2">Sarcasm Detection Study</h1>
      <p className="text-gray-600 mb-8">EECS 543 - AI Ethics Research</p>

      <div className="bg-white rounded-xl p-6 shadow-sm">
        <h2 className="text-lg font-semibold mb-4">Welcome, Researcher</h2>
        <p className="text-gray-600 mb-6">
          This study examines human ability to detect sarcasm in Reddit conversations,
          with and without AI assistance.
        </p>

        <div className="space-y-3">
          <Link
            href="/no-ai"
            className="block w-full p-4 bg-gray-100 hover:bg-gray-200 rounded-lg transition"
          >
            <span className="font-medium">No-AI Condition</span>
            <span className="block text-sm text-gray-500">Baseline - no AI assistance</span>
          </Link>

          <Link
            href="/with-ai"
            className="block w-full p-4 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition"
          >
            <span className="font-medium">With-AI Condition</span>
            <span className="block text-sm text-blue-100">AI predictions shown</span>
          </Link>
        </div>

        <p className="text-sm text-gray-500 mt-6 italic">
          For MTurk deployment, share the specific condition URL with participants.
        </p>
      </div>
    </main>
  )
}
