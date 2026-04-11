import os
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

from bitgn.harness_connect import HarnessServiceClientSync
from bitgn.harness_pb2 import EndTrialRequest, SubmitRunRequest, EvalPolicy, StartTrialRequest, GetBenchmarkRequest, GetRunRequest, StatusRequest, StartRunRequest
from connectrpc.errors import ConnectError
from openai import OpenAI

from agent import run_agent
from logging_utils import configure_logging

BITGN_URL = os.getenv("BITGN_HOST") or "https://api.bitgn.com"
BITGN_API_KEY = os.getenv("BITGN_API_KEY") or ""
BENCH_ID = os.getenv("BENCH_ID") or "bitgn/pac1-prod"
MODEL_ID = os.getenv("MODEL_ID") or "gpt-4.1-2025-04-14"
MAX_WORKERS = max(1, int(os.getenv("PAC1_MAX_WORKERS") or "4"))
TASK_DELAY_SEC = max(0.0, float(os.getenv("PAC1_TASK_DELAY_SEC") or "5"))

CLI_RED = "\x1B[31m"
CLI_GREEN = "\x1B[32m"
CLI_CLR = "\x1B[0m"
CLI_BLUE = "\x1B[34m"


def _run_trial(trial_id: str, task_id: str) -> tuple[str, float] | None:
    logger, _ = configure_logging()
    client = HarnessServiceClientSync(BITGN_URL)
    trial = client.start_trial(StartTrialRequest(trial_id=trial_id))

    logger.info(f"{'=' * 30} Starting task: {task_id} {'=' * 30}")
    logger.info(f"{CLI_BLUE}{trial.instruction}{CLI_CLR}\n{'-' * 80}")

    try:
        run_agent(MODEL_ID, trial.harness_url, trial.instruction)
    except Exception as exc:
        logger.exception("Agent execution failed for %s: %s", task_id, exc)

    result = client.end_trial(EndTrialRequest(trial_id=trial.trial_id))
    if result.score >= 0:
        style = CLI_GREEN if result.score == 1 else CLI_RED
        explain = textwrap.indent("\n".join(result.score_detail), "  ")
        logger.info(
            f"\n{style}Score for {task_id}: {result.score:0.2f}\n{explain}\n{CLI_CLR}"
        )
        if TASK_DELAY_SEC > 0:
            logger.info("Sleeping %.1f seconds before the next task.", TASK_DELAY_SEC)
            time.sleep(TASK_DELAY_SEC)
        return task_id, result.score
    if TASK_DELAY_SEC > 0:
        logger.info("Sleeping %.1f seconds before the next task.", TASK_DELAY_SEC)
        time.sleep(TASK_DELAY_SEC)
    return None


def _verify_model_call() -> None:
    logger, _ = configure_logging()
    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY") or "not-needed",
    )
    logger.info("Checking model call for `%s`...", MODEL_ID)
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": "Reply with OK."},
            {"role": "user", "content": "Ping"},
        ],
        max_completion_tokens=16,
    )
    content = resp.choices[0].message.content or ""
    if not content.strip():
        raise RuntimeError(f"Model `{MODEL_ID}` returned an empty response during preflight.")
    logger.info("Model preflight OK: %s", content.strip())


def main() -> None:
    logger, log_path = configure_logging()
    task_filter = set(os.sys.argv[1:])

    scores = []
    try:
        client = HarnessServiceClientSync(BITGN_URL)

        logger.info("Writing logs to %s", log_path)
        logger.info("Connecting to BitGN %s", client.status(StatusRequest()))
        res = client.get_benchmark(GetBenchmarkRequest(benchmark_id=BENCH_ID))
        logger.info(
            f"{EvalPolicy.Name(res.policy)} benchmark: {res.benchmark_id} "
            f"with {len(res.tasks)} tasks.\n{CLI_GREEN}{res.description}{CLI_CLR}"
        )
        _verify_model_call()

        run = client.start_run(StartRunRequest(
            name="Ho Dzha",
            benchmark_id=BENCH_ID,
            api_key=BITGN_API_KEY))

        try:
            run_state = client.get_run(GetRunRequest(run_id=run.run_id))
            selected_trials = [
                (trial.trial_id, trial.task_id)
                for trial in run_state.trials
                if not task_filter or trial.task_id in task_filter
            ]

            if not selected_trials:
                logger.warning("No tasks matched the provided filter: %s", sorted(task_filter))
            else:
                worker_count = min(MAX_WORKERS, len(selected_trials))
                logger.info(
                    "Running %s task(s) with up to %s worker thread(s).",
                    len(selected_trials),
                    worker_count,
                )

                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures = [
                        executor.submit(_run_trial, trial_id, task_id)
                        for trial_id, task_id in selected_trials
                    ]
                    for future in as_completed(futures):
                        outcome = future.result()
                        if outcome is not None:
                            scores.append(outcome)

        finally:
            client.submit_run(SubmitRunRequest(run_id=run.run_id, force=True))

    except ConnectError as exc:
        logger.error("%s: %s", exc.code, exc.message)
    except KeyboardInterrupt:
        logger.warning(f"{CLI_RED}Interrupted{CLI_CLR}")

    if scores:
        for task_id, score in scores:
            style = CLI_GREEN if score == 1 else CLI_RED
            logger.info(f"{task_id}: {style}{score:0.2f}{CLI_CLR}")

        total = sum(score for _, score in scores) / len(scores) * 100.0
        logger.info(f"FINAL: {total:0.2f}%")


if __name__ == "__main__":
    main()
