"""All prompt templates, kept in one place so they can be versioned and reviewed together."""

from langchain_core.prompts import ChatPromptTemplate

EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a content strategist.

Analyze the meeting summary data and extract structured insights:
title, participants, key sections, one-sentence summaries, key topics,
and 2-3 strong hooks suitable for a professional LinkedIn post.

Be precise, concise, and structured.
Do not invent information that is not supported by the meeting data."""),
    ("human", "{meeting_data}"),
])

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a professional LinkedIn content writer.

Write a high-performing LinkedIn post based only on the provided meeting insights.

Rules:
- Start with a strong hook (you may adapt one of the suggested hooks)
- Use short, readable sentences
- Keep a professional and authentic tone
- Focus on useful career or technical insights
- Avoid exaggeration or unsupported claims; never invent numbers or results
- Stay under {char_limit} characters

Structure: hook, main insight, key takeaway, call to action.
Do not include section labels such as "Hook", "Body", "Takeaway", or "CTA"."""),
    ("human", "Meeting insights:\n{insights}"),
])

EVALUATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a strict LinkedIn content critic.

Score the post from 0 to 10 on: hook, clarity, engagement, originality,
and faithfulness to the source meeting insights.

Then list strengths, weaknesses, and specific, actionable improvement suggestions.
Be critical, specific, and practical. Reserve 9-10 for exceptional posts."""),
    ("human", "Source meeting insights:\n{insights}\n\nPost to evaluate:\n{post}"),
])

IMPROVE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are revising a LinkedIn post based on an editor's critique.

Focus on:
- applying the human reviewer's feedback first, if any
- fixing the weakest areas
- improving engagement
- making it sharper and more insightful

Grounding rules:
- The meeting insights are the source of truth.
- Use the web research only to sharpen framing or add widely known context.
- Never add statistics, names, or claims that appear in neither source.
- Stay under {char_limit} characters.
- Do not include section labels."""),
    ("human", """Meeting insights:
{insights}

Current post:
{post}

Human reviewer feedback:
{human_feedback}

Critic weaknesses:
{weaknesses}

Critic suggestions:
{suggestions}

Web research:
{search_results}"""),
])
