import { NextRequest, NextResponse } from 'next/server'
import { createServiceClient } from '@/lib/supabase'
import { generateCompletionCode } from '@/lib/trials'

export async function POST(request: NextRequest) {
  try {
    const { worker_id, condition, responses } = await request.json()

    if (!worker_id || !condition || !responses) {
      return NextResponse.json({ error: 'Invalid data' }, { status: 400 })
    }

    const completionCode = generateCompletionCode()
    const correct = responses.filter(
      (r: { participant_answer: string; ground_truth: string }) =>
        r.participant_answer === r.ground_truth
    ).length
    const accuracy = correct / responses.length

    // Save session to Supabase
    const supabase = createServiceClient()
    await supabase.from('sessions').insert({
      worker_id,
      condition,
      completion_code: completionCode,
      trials_completed: responses.length,
      accuracy
    })

    return NextResponse.json({
      success: true,
      completion_code: completionCode
    })
  } catch (err) {
    console.error('Submit error:', err)
    return NextResponse.json({ error: 'Server error' }, { status: 500 })
  }
}
