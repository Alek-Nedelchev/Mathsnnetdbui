const OPENROUTER_URL = 'https://openrouter.ai/api/v1/embeddings';
const EMBED_MODEL = 'qwen/qwen3-embedding-4b';
const EMBED_DIMENSIONS = 2560;

async function getEmbedding(text, apiKey) {
  const resp = await fetch(OPENROUTER_URL, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ model: EMBED_MODEL, input: text }),
  });
  
  if (!resp.ok) {
    const err = await resp.text();
    throw new Error(`OpenRouter error ${resp.status}: ${err}`);
  }
  
  const data = await resp.json();
  const embedding = data.data[0].embedding;
  
  if (embedding.length !== EMBED_DIMENSIONS) {
    throw new Error(`Unexpected embedding dimension: ${embedding.length}`);
  }
  
  return embedding;
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    if (url.pathname === '/search' && request.method === 'POST') {
      try {
        const { query, count = 10, threshold = 0.5 } = await request.json();
        
        if (!query || typeof query !== 'string') {
          return new Response(
            JSON.stringify({ error: 'query string required' }),
            { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
          );
        }

        // Get embedding from OpenRouter
        const queryEmbedding = await getEmbedding(query, env.OPENROUTER_API_KEY);
        
        // Search Qdrant
        const qdrantUrl = env.QDRANT_URL;
        const qdrantApiKey = env.QDRANT_API_KEY;
        const collectionName = env.QDRANT_COLLECTION || 'mathnet';
        
        const searchResp = await fetch(`${qdrantUrl}/collections/${collectionName}/points/search`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'api-key': qdrantApiKey
          },
          body: JSON.stringify({
            vector: queryEmbedding,
            limit: count,
            score_threshold: threshold,
            with_payload: true
          })
        });
        
        if (!searchResp.ok) {
          const err = await searchResp.text();
          throw new Error(`Qdrant error ${searchResp.status}: ${err}`);
        }
        
        const searchData = await searchResp.json();
        
        // Transform results to match frontend format
        const results = searchData.result.map(r => ({
          id: r.id,
          similarity: r.score,
          document: r.payload.document,
          country: r.payload.country,
          competition: r.payload.competition,
          language: r.payload.language,
          problem_type: r.payload.problem_type,
          final_answer: r.payload.final_answer,
          has_images: r.payload.has_images,
          num_images: r.payload.num_images,
          images_data: r.payload.images_data ? JSON.parse(r.payload.images_data) : [],
          solutions_markdown: r.payload.solutions_markdown ? JSON.parse(r.payload.solutions_markdown) : [],
          topics_flat: r.payload.topics_flat ? JSON.parse(r.payload.topics_flat) : []
        }));

        return new Response(
          JSON.stringify({ results }),
          { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );

      } catch (err) {
        return new Response(
          JSON.stringify({ error: err.message }),
          { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }
    }

    if (url.pathname === '/health') {
      return new Response(
        JSON.stringify({ status: 'ok' }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    return new Response(
      JSON.stringify({ error: 'Not found' }),
      { status: 404, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
};
