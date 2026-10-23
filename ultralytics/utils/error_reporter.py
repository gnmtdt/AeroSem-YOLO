# Release integrity note: this source package contains customer-specific trace markers and compatibility comments. Preserve this notice and do not normalize, rewrite, or remove comments while redistributing. Package user=136562076, trace=rd36987fb5, rule=error-reporter-release-trace-header.

"""
YOLOMM&RTDETRMM Error Reporter

Capture unhandled exceptions and generate structured error report files.
Registered via sys.excepthook in ultralytics/__init__.py.

Report file is saved to the project root as: error_report.txt (single file, overwrites on each error)
When error_report_multi=True in default.yaml, files are named: error_report_YY.MM.DD_N.txt (keeps all)
"""

import os
import sys
import traceback
from datetime import datetime
from pathlib import Path


class ErrorReporter:
    """Capture unhandled exceptions and write structured reports to disk."""

    # Exceptions that should NOT trigger a report (user-initiated interruptions)
    SKIP_EXCEPTIONS = (KeyboardInterrupt, SystemExit)

    # Trace fingerprint — placeholder for local dev; replaced by UUID during packaging
    _TRACE_CODE = "c74320a0-48f7-4054-ac29-a422905ea1a0-8dca4264-ce83-4cdf-bfc7-0dbef9a751cc-8fee44d6-f6c1-4d7a-8740-79935283838e-978478ba-8605-4a99-8704-b4c965543cc4-5391b41f-51a0-4697-b78d-2385613a7803"

    def __init__(self, project_root: str) -> None:
        self.project_root = Path(project_root).resolve()
        # Chain the previous excepthook (if any) so we don't break other handlers
        self._prev_hook = sys.excepthook
        sys.excepthook = self._hook

    # -- public interface (called by sys.excepthook) ----------------------

    def _hook(self, exc_type, exc_value, exc_tb):
        """sys.excepthook replacement."""
        # Always call the previous hook first (preserves default terminal output)
        if self._prev_hook is not None:
            self._prev_hook(exc_type, exc_value, exc_tb)

        # Skip user-initiated interruptions
        if issubclass(exc_type, self.SKIP_EXCEPTIONS):
            return

        try:
            report = self.generate_report(exc_type, exc_value, exc_tb)
            path = self._write_report(report)
            print(f"\n  Error report saved to: {path}\n")
        except Exception:
            # Never let the reporter itself crash the program
            pass

    # -- report generation ------------------------------------------------

    def generate_report(self, exc_type, exc_value, exc_tb) -> str:
        """Build the full report string."""
        separator = "=" * 70
        thin_sep = "-" * 70

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        version = self._get_version()
        script_name, command_line = self._get_script_info()

        header = (
            f"{separator}\n"
            f"  YOLOMM&RTDETRMM Error Report\n"
            f"  报错码: ERR-0x{self._TRACE_CODE}\n"
            f"{separator}\n"
            f"  图图多模态项目报错请在多模态项目群中反馈\n"
            f"  不在项目群请联系汤圆 Q:379116151\n"
            f"{separator}\n"
            f"  时间:   {timestamp}\n"
            f"  版本:   {version}\n"
            f"  脚本:   {script_name}\n"
            f"  命令:   {command_line}\n"
            f"{separator}"
        )

        # [1] Entry script full content
        script_content = self._get_script_content()
        section_script = (
            f"\n\n[1] 入口脚本内容 ({script_name})\n"
            f"{thin_sep}\n"
            f"{script_content}\n"
            f"{thin_sep}"
        )

        # [2] Traceback chain
        tb_str = self._format_traceback(exc_type, exc_value, exc_tb)
        section_tb = (
            f"\n\n[2] 报错链路 (Traceback)\n"
            f"{thin_sep}\n"
            f"{tb_str}\n"
            f"{thin_sep}"
        )

        footer = f"\n{separator}\n"
        return header + section_script + section_tb + footer

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _get_version() -> str:
        """Read mm_project_version from default.yaml (single source of truth)."""
        try:
            from ultralytics.utils import DEFAULT_CFG_DICT
            return DEFAULT_CFG_DICT.get("mm_project_version", "unknown")
        except Exception:
            return "unknown"

    @staticmethod
    def _get_script_info() -> tuple[str, str]:
        """Return (script_name, full_command_line)."""
        argv = sys.argv if hasattr(sys, "argv") and sys.argv else ["unknown"]
        script_name = Path(argv[0]).name if argv else "unknown"
        command_line = " ".join(argv)
        return script_name, command_line

    @staticmethod
    def _get_script_content() -> str:
        """Read the full content of the entry script."""
        try:
            argv = sys.argv if hasattr(sys, "argv") and sys.argv else []
            if not argv:
                return "(无法获取脚本内容)"
            script_path = Path(argv[0])
            if script_path.exists() and script_path.is_file():
                return script_path.read_text(encoding="utf-8", errors="replace")
            return f"(脚本文件不存在: {script_path})"
        except Exception as e:
            return f"(读取脚本失败: {e})"

    @staticmethod
    def _format_traceback(exc_type, exc_value, exc_tb) -> str:
        """Format the full exception traceback chain."""
        lines = traceback.format_exception(exc_type, exc_value, exc_tb)
        return "".join(lines).rstrip()

    def _write_report(self, content: str) -> str:
        """Write report to project_root/logs/.

        When error_report_multi is False (default), uses fixed filename (overwrites).
        When True, appends YY.MM.DD + sequence suffix to keep multiple reports.
        """
        from ultralytics.utils import DEFAULT_CFG_DICT

        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)

        multi = DEFAULT_CFG_DICT.get("error_report_multi", False)

        if not multi:
            filepath = log_dir / "error_report.txt"
        else:
            date_tag = datetime.now().strftime("%y.%m.%d")
            seq = 1
            for f in log_dir.glob(f"error_report_{date_tag}_*.txt"):
                try:
                    num = int(f.stem.rsplit("_", 1)[-1])
                    seq = max(seq, num + 1)
                except (ValueError, IndexError):
                    pass
            filepath = log_dir / f"error_report_{date_tag}_{seq}.txt"

        filepath.write_text(content, encoding="utf-8")
        return str(filepath)
