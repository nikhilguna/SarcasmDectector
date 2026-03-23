import { NextRequest, NextResponse } from 'next/server'
import { getTrialsForWorker, generateWorkerId } from '@/lib/trials'

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const condition = searchParams.get('condition') as 'no-ai' | 'with-ai'

  const workerId = generateWorkerId()
  const includeAI = condition === 'with-ai'
  const trials = getTrialsForWorker(workerId, includeAI)

  return NextResponse.json({
    worker_id: workerId,
    condition,
    trials,
    total: trials.length
  })
}
