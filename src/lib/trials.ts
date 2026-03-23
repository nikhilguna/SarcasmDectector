import trialsData from '../../model_outputs.json'

export interface Trial {
  trial_id: number
  context: string[]
  response: string
  ground_truth: string
  model_prediction: string
  model_confidence: number
  model_reasoning: string
}

const NUM_TRIALS = 20

// Seeded random number generator (mulberry32)
function seededRandom(seed: number): number {
  let t = seed + 0x6D2B79F5
  t = Math.imul(t ^ t >>> 15, t | 1)
  t ^= t + Math.imul(t ^ t >>> 7, t | 61)
  return ((t ^ t >>> 14) >>> 0) / 4294967296
}

function hashString(str: string): number {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i)
    hash = ((hash << 5) - hash) + char
    hash = hash & hash
  }
  return Math.abs(hash)
}

// Fisher-Yates shuffle with seed
function seededShuffle<T>(array: T[], seed: number): T[] {
  const result = [...array]
  let currentSeed = seed

  for (let i = result.length - 1; i > 0; i--) {
    currentSeed = hashString(currentSeed.toString() + i)
    const rand = seededRandom(currentSeed)
    const j = Math.floor(rand * (i + 1))
    ;[result[i], result[j]] = [result[j], result[i]]
  }
  return result
}

export function getTrialsForWorker(workerId: string, includeAI: boolean): Trial[] {
  const seed = hashString(workerId)
  const shuffled = seededShuffle(trialsData as Trial[], seed)
  const selected = shuffled.slice(0, NUM_TRIALS)

  if (!includeAI) {
    // Remove AI fields for no-ai condition
    return selected.map(t => ({
      trial_id: t.trial_id,
      context: t.context,
      response: t.response,
      ground_truth: t.ground_truth,
      model_prediction: '',
      model_confidence: 0,
      model_reasoning: ''
    }))
  }

  return selected
}

export function generateWorkerId(): string {
  const timestamp = Date.now().toString(36)
  const random = Math.random().toString(36).substring(2, 8)
  return `P-${timestamp}-${random}`.toUpperCase()
}

export function generateCompletionCode(): string {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
  let code = 'SARC-'
  for (let i = 0; i < 4; i++) {
    code += chars.charAt(Math.floor(Math.random() * chars.length))
  }
  return code
}
