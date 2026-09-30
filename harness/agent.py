import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from harness.tools import registry

# Load API keys from .env into the environment
# override=True ensures a real key in .env wins over a stale/placeholder
# key that may already be set in the shell/system environment.
load_dotenv(override=True)

# Pick the provider with LLM_PROVIDER=kimi or LLM_PROVIDER=openai (default: openai)
PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

if PROVIDER == "kimi":
    MODEL = os.getenv("KIMI_MODEL", "kimi-k3")
    client = OpenAI(
        api_key=os.getenv("KIMI_API_KEY"),
        base_url="https://api.moonshot.ai/v1",
    )
    EXTRA_BODY = {"thinking": {"type": "disabled"}} 
else:
    MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    EXTRA_BODY = {}


SYSTEM_PROMPT = """ You are a coding assistant running in a terminal, helping a developer with software engineering tasks.

Be concise. Prefer short, direct answers over long ones. When the user asks for code, return the code with minimal explanation unless they ask for more.

When returning code, use fenced code blocks and specify the language.

You have access to five filesystem tools — read, write, list, mkdir, delete — operating on a workspace directory. Use them whenever a task involves reading, modifying, or organizing files. Paths are relative to
the workspace root. Prefer reading and writing real files over describing them in conversation.

You also have eight git tools — git_status, git_diff, git_log, git_commit, git_checkout, git_branch, git_remote, git_push — for versioning your work. The workspace is
already initialized as a git repo. Use git to:
- Commit frequently. Small, focused commits are easier to roll back.
- Commit before doing anything risky (large rewrites, deleting files, restructuring). A commit before the risky step gives you a recovery point.
- Write meaningful commit messages — describe what changed and why, in the present tense (e.g., "add user authentication module").
- Branch experiments. When trying an alternative approach, create a branch first so the main line of work stays intact.
- Before pushing, confirm a remote named "origin" is configured (git_remote). Push only the current branch (git_push) — force-pushing is not available.
"""

def run():
    """Run the agent's conversation loop until the user quits."""

    # The conversation history. This is the entire memory of the agent.
    # Every turn, we append to it and send the whole thing to the model.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Agent ready. Type 'quit' or 'exit' to leave.\n")

    while True:
        # 1. Get input from the user
        user_input = input("you > ").strip()

        # 2. Allow the user to leave cleanly
        if user_input in {"quit", "exit"}:
            print("Goodbye.")
            break

        # Skip empty lines without making a model call
        if not user_input:
            continue

        # 3. Append the user's message to the history
        messages.append({"role": "user", "content": user_input})

        # 4. Call the model with the full conversation so far and available tools. 
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            extra_body=EXTRA_BODY,
            tools=registry.get_schemas()
        )
        
        message = response.choices[0].message                              
        # If the model asked for a tool call, handle it before producing    
        # the user-facing reply. Minimum-viable dispatch: one round only.
        if message.tool_calls:
            # Step 1: record the model's tool-call message in history so the
            # upcoming tool-result messages have something to reference.
            messages.append(message)

            # Step 2: run each requested tool and append its result to history,
            # using the matching tool_call_id so the model can pair them up.
            for call in message.tool_calls:
                arguments = json.loads(call.function.arguments)
                result = registry.dispatch(call.function.name, arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })

            # Step 3: re-call the model now that the tool results are in
            # context. This second call produces the model's final text reply.
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=registry.get_schemas(),
            )
            message = response.choices[0].message

        # By here, `message` is the model's final text response for this turn —
        # either from the first call (no tools needed) or the second (after dispatch).
        assistant_text = message.content or "(no text response - used tools only)"
        messages.append({"role": "assistant", "content": assistant_text})

        print(f"\nagent > {assistant_text}\n")

       


if __name__ == "__main__":
    run()