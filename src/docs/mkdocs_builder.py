import logging
import re
import select
import subprocess
import sys
import termios
import threading
import tty
from pathlib import Path

import yaml

from src.config import RunConfig

logger = logging.getLogger(__name__)


class MkDocsBuilder:
    def __init__(self, config: RunConfig, project_dir: str):
        self.config = config
        self.project_dir = Path(project_dir).resolve()
        self.docs_dir = self.project_dir / "docs"
        self.site_dir = self.project_dir / "site"
        self.mkdocs_path = self.project_dir / "mkdocs.yml"
        self.classified_path = self.project_dir / "manifest" / "classified_files.json"
        self.file_summaries_dir = self.project_dir / "summaries" / "files"
        self.module_summaries_dir = self.project_dir / "summaries" / "modules"

    def run(self) -> bool:
        if not self.docs_dir.exists():
            logger.error("Step 10 aborted: docs directory not found at %s", self.docs_dir)
            return False

        inventory = self._collect_inventory()
        nav = self._build_nav(inventory)
        config = self._build_mkdocs_config(nav)
        self._write_mkdocs_yml(config)
        self._write_extra_assets()

        if not self._build_site():
            return False

        if not self._verify_site():
            return False

        if self.config.serve_site:
            return self._serve_site()

        return True

    def _collect_inventory(self) -> dict:
        classified = {}
        if self.classified_path.exists():
            classified = yaml.safe_load(self.classified_path.read_text(encoding="utf-8")) or {}

        file_entries = []
        for doc_file in sorted((self.docs_dir / "files").glob("*.md")):
            summary = self._load_yaml_or_json(self.file_summaries_dir / f"{doc_file.stem}.json")
            source_path = summary.get("file") if isinstance(summary, dict) else None
            if not source_path:
                source_path = doc_file.stem.replace("__", "/")

            is_test = bool(classified.get(source_path, {}).get("is_test", False))
            file_entries.append(
                {
                    "label": Path(source_path).name,
                    "doc_path": f"files/{doc_file.name}",
                    "source_path": source_path,
                    "group": str(Path(source_path).parent),
                    "is_test": is_test,
                }
            )

        module_entries = []
        for doc_file in sorted((self.docs_dir / "modules").glob("*.md")):
            summary = self._load_yaml_or_json(self.module_summaries_dir / f"{doc_file.stem}.json")
            folder_path = summary.get("folder") if isinstance(summary, dict) else None
            if not folder_path:
                folder_path = doc_file.stem.replace("__", "/")

            is_test = "/tests" in f"/{folder_path}/" or folder_path.startswith("tests")
            module_entries.append(
                {
                    "label": folder_path,
                    "doc_path": f"modules/{doc_file.name}",
                    "folder_path": folder_path,
                    "is_test": is_test,
                }
            )

        diagram_entries = []
        diagrams_dir = self.docs_dir / "diagrams"
        if diagrams_dir.exists():
            for doc_file in sorted(diagrams_dir.glob("*.md")):
                name_map = {
                    "component_graph.md": "Component Graph",
                    "import_graph_top30.md": "Import Graph Top 30",
                }
                diagram_entries.append(
                    {
                        "label": name_map.get(doc_file.name, doc_file.stem.replace("_", " ").title()),
                        "doc_path": f"diagrams/{doc_file.name}",
                    }
                )

        return {
            "file_entries": self._dedupe_file_labels(file_entries),
            "module_entries": module_entries,
            "diagram_entries": diagram_entries,
            "has_index": (self.docs_dir / "index.md").exists(),
            "has_setup": (self.docs_dir / "setup.md").exists(),
            "has_architecture": (self.docs_dir / "architecture.md").exists(),
            "has_reference": (self.docs_dir / "reference.md").exists(),
        }

    def _build_nav(self, inventory: dict) -> list:
        nav: list = []

        if inventory["has_index"]:
            nav.append({"Home": "index.md"})
        if inventory["has_setup"]:
            nav.append({"Setup": "setup.md"})
        if inventory["has_architecture"]:
            nav.append({"Architecture": "architecture.md"})

        prod_modules = [m for m in inventory["module_entries"] if not m["is_test"]]
        test_modules = [m for m in inventory["module_entries"] if m["is_test"]]
        prod_files = [f for f in inventory["file_entries"] if not f["is_test"]]
        test_files = [f for f in inventory["file_entries"] if f["is_test"]]

        if prod_modules:
            nav.append({"Modules": [{m["label"]: m["doc_path"]} for m in sorted(prod_modules, key=lambda x: x["label"]) ]})

        if prod_files:
            nav.append({"Files": self._group_files(prod_files)})

        tests_nav = []
        if test_modules:
            tests_nav.append({"Modules": [{m["label"]: m["doc_path"]} for m in sorted(test_modules, key=lambda x: x["label"]) ]})
        if test_files:
            tests_nav.append({"Files": self._group_files(test_files)})
        if tests_nav:
            nav.append({"Tests": tests_nav})

        if inventory["diagram_entries"]:
            nav.append(
                {
                    "Diagrams": [
                        {entry["label"]: entry["doc_path"]}
                        for entry in inventory["diagram_entries"]
                    ]
                }
            )

        if inventory["has_reference"]:
            nav.append({"Reference": "reference.md"})

        return nav

    def _group_files(self, file_entries: list[dict]) -> list:
        grouped: dict[str, list[dict]] = {}
        for entry in file_entries:
            group_key = entry["group"] if entry["group"] != "." else "root"
            grouped.setdefault(group_key, []).append(entry)

        nav_groups = []
        for group in sorted(grouped.keys()):
            entries = sorted(grouped[group], key=lambda x: (x["label"], x["source_path"]))
            group_label = "root" if group == "root" else group
            nav_groups.append({group_label: [{item["label"]: item["doc_path"]} for item in entries]})
        return nav_groups

    def _build_mkdocs_config(self, nav: list) -> dict:
        return {
            "site_name": f"{self.project_dir.name} Documentation",
            "docs_dir": "docs",
            "site_dir": "site",
            "theme": {
                "name": "material",
                "features": [
                    "navigation.tabs",
                    "navigation.top",
                    "search.suggest",
                    "search.highlight",
                ],
            },
            "plugins": ["search", "mermaid2", "panzoom"],
            "extra_css": ["css/custom.css"],
            "markdown_extensions": [
                "admonition",
                "tables",
                "toc",
                "pymdownx.details",
                {
                    "pymdownx.superfences": {
                        "custom_fences": [
                            {
                                "name": "mermaid",
                                "class": "mermaid",
                                "format": "__MERMAID_CUSTOM_FENCE__",
                            }
                        ]
                    }
                },
                "pymdownx.highlight",
                "pymdownx.blocks.admonition",
            ],
            "nav": nav,
        }

    def _write_mkdocs_yml(self, config: dict) -> None:
        text = yaml.safe_dump(config, sort_keys=False, allow_unicode=False)
        text = text.replace(
            "'__MERMAID_CUSTOM_FENCE__'",
            "!!python/name:mermaid2.fence_mermaid_custom",
        )
        text = text.replace(
            '"__MERMAID_CUSTOM_FENCE__"',
            "!!python/name:mermaid2.fence_mermaid_custom",
        )
        text = text.replace(
            "__MERMAID_CUSTOM_FENCE__",
            "!!python/name:mermaid2.fence_mermaid_custom",
        )
        self.mkdocs_path.write_text(text, encoding="utf-8")
        logger.info("Wrote MkDocs config: %s", self.mkdocs_path)

    def _build_site(self) -> bool:
        logger.info("Running Step 10 build: mkdocs build")
        commands = self._mkdocs_commands("build")

        completed = None
        for command in commands:
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.project_dir,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                break
            except FileNotFoundError:
                continue
            except Exception as exc:
                logger.error("Failed to execute mkdocs build command %s: %s", command, exc)
                return False

        if completed is None:
            logger.error(
                "mkdocs build failed: neither 'mkdocs' nor 'python -m mkdocs' could be executed."
            )
            return False

        if completed.returncode != 0:
            logger.error("mkdocs build failed with exit code %s", completed.returncode)
            if completed.stdout:
                logger.error("mkdocs stdout:\n%s", completed.stdout)
            if completed.stderr:
                logger.error("mkdocs stderr:\n%s", completed.stderr)
            return False

        if completed.stdout:
            logger.info("mkdocs build output:\n%s", completed.stdout)
        return True

    def _mkdocs_commands(self, subcommand: str) -> list[list[str]]:
        base_args = [subcommand, "-f", str(self.mkdocs_path)]
        if subcommand == "build":
            base_args.insert(1, "--clean")
        if subcommand == "serve":
            base_args.extend(["-a", f"{self.config.serve_host}:{self.config.serve_port}"])

        return [
            ["mkdocs", *base_args],
            [sys.executable, "-m", "mkdocs", *base_args],
        ]

    def _serve_site(self) -> bool:
        logger.info("Starting Step 10 local hosting: mkdocs serve")
        serve_commands = self._mkdocs_commands("serve")

        process = None
        for command in serve_commands:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=self.project_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                break
            except FileNotFoundError:
                continue
            except Exception as exc:
                logger.error("Failed to execute mkdocs serve command %s: %s", command, exc)
                return False

        if process is None:
            logger.error(
                "mkdocs serve failed: neither 'mkdocs' nor 'python -m mkdocs' could be executed."
            )
            return False

        default_url = f"http://{self.config.serve_host}:{self.config.serve_port}/"
        shared = {"url": default_url}
        url_ready = threading.Event()

        def _pump_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                message = line.rstrip()
                if message:
                    logger.info("[mkdocs-serve] %s", message)
                if not url_ready.is_set():
                    match = re.search(r"http://[^\s]+", message)
                    if match:
                        shared["url"] = match.group(0)
                        url_ready.set()
            if not url_ready.is_set():
                url_ready.set()

        output_thread = threading.Thread(target=_pump_output, daemon=True)
        output_thread.start()
        url_ready.wait(timeout=5)

        logger.info("Live documentation URL: %s", shared["url"])
        logger.info("Server is running. Press Ctrl+G to stop the server and exit.")

        return self._wait_for_ctrl_g(process)

    def _wait_for_ctrl_g(self, process: subprocess.Popen) -> bool:
        if not sys.stdin.isatty():
            logger.warning(
                "Interactive terminal not detected. Ctrl+G capture disabled; press Ctrl+C to stop the server."
            )
            try:
                process.wait()
            except KeyboardInterrupt:
                pass
            finally:
                self._terminate_process(process)
            return True

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)
            while True:
                if process.poll() is not None:
                    logger.error("mkdocs serve exited unexpectedly with code %s", process.returncode)
                    return False

                readable, _, _ = select.select([sys.stdin], [], [], 0.2)
                if not readable:
                    continue

                key = sys.stdin.read(1)
                if key == "\x07":  # Ctrl+G
                    logger.info("Ctrl+G received. Stopping local docs server...")
                    self._terminate_process(process)
                    return True

                if key == "\x03":  # Ctrl+C
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Stopping local docs server...")
            self._terminate_process(process)
            return True
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    @staticmethod
    def _terminate_process(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return

        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def _verify_site(self) -> bool:
        index_path = self.site_dir / "index.html"
        if not index_path.exists():
            logger.error("Step 10 verification failed: %s missing", index_path)
            return False

        logger.info("Step 10 complete. Static site available at: %s", self.site_dir)
        return True

    def _dedupe_file_labels(self, file_entries: list[dict]) -> list[dict]:
        grouped_counts: dict[tuple[str, str], int] = {}
        for entry in file_entries:
            key = (entry["group"], entry["label"])
            grouped_counts[key] = grouped_counts.get(key, 0) + 1

        deduped = []
        for entry in file_entries:
            key = (entry["group"], entry["label"])
            if grouped_counts[key] > 1:
                parent = Path(entry["source_path"]).parent.name or "root"
                entry = dict(entry)
                entry["label"] = f"{entry['label']} ({parent})"
            deduped.append(entry)
        return deduped

    def _write_extra_assets(self) -> None:
        """Writes custom CSS and potentially other static assets to docs/."""
        css_dir = self.docs_dir / "css"
        css_dir.mkdir(parents=True, exist_ok=True)
        
        css_path = css_dir / "custom.css"
        css_content = (
            "/* Expand Mermaid viewports for better pan/zoom interaction */\n"
            ".mermaid {\n"
            "    min-height: 480px;\n"
            "    width: 100%;\n"
            "    display: block;\n"
            "    margin-bottom: 2em;\n"
            "}\n\n"
            "/* Targeted height for complex import graphs */\n"
            "[id*='import_graph'] .mermaid {\n"
            "    min-height: 650px;\n"
            "}\n"
        )
        css_path.write_text(css_content, encoding="utf-8")
        logger.info("Wrote extra assets (CSS) to %s", css_path)

    @staticmethod
    def _load_yaml_or_json(path: Path):
        if not path.exists():
            return {}
        try:
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}


def run_step_10(config: RunConfig, project_dir: str) -> bool:
    builder = MkDocsBuilder(config, project_dir)
    return builder.run()
