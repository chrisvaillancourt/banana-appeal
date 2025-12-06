"""Tests to verify the package can be built and installed correctly."""

import subprocess
import sys
from pathlib import Path

import pytest


class TestPackageMetadata:
    """Tests for package metadata and imports."""

    def test_package_importable(self):
        """Test that the package can be imported."""
        import banana_appeal

        assert banana_appeal is not None

    def test_version_defined(self):
        """Test that version is defined and valid."""
        from banana_appeal import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)
        # Should be semver format
        parts = __version__.split(".")
        assert len(parts) >= 2

    def test_server_module_importable(self):
        """Test that the server module can be imported."""
        from banana_appeal import server

        assert server is not None
        assert hasattr(server, "mcp")
        assert hasattr(server, "main")

    def test_models_module_importable(self):
        """Test that the models module can be imported."""
        from banana_appeal import models

        assert models is not None
        assert hasattr(models, "GenerateImageRequest")
        assert hasattr(models, "EditImageRequest")
        assert hasattr(models, "BlendImagesRequest")

    def test_mcp_tools_registered(self):
        """Test that MCP tools are properly registered."""
        from banana_appeal.server import mcp

        # The FastMCP instance should have tools registered
        assert mcp is not None
        assert mcp.name == "banana-appeal"


class TestPackageBuild:
    """Tests for package building."""

    @pytest.fixture
    def project_root(self):
        """Get the project root directory."""
        return Path(__file__).parent.parent

    def test_pyproject_toml_valid(self, project_root):
        """Test that pyproject.toml is valid."""
        pyproject = project_root / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml not found"

        # Try to parse it
        try:
            import tomllib

            with pyproject.open("rb") as f:
                data = tomllib.load(f)
        except ImportError:
            import tomli as tomllib

            with pyproject.open("rb") as f:
                data = tomllib.load(f)

        # Check required fields
        assert "project" in data
        assert "name" in data["project"]
        assert "version" in data["project"]
        assert data["project"]["name"] == "banana-appeal"

    def test_readme_exists(self, project_root):
        """Test that README.md exists."""
        readme = project_root / "README.md"
        assert readme.exists(), "README.md not found"
        content = readme.read_text()
        assert len(content) > 100, "README.md seems too short"

    def test_license_exists(self, project_root):
        """Test that LICENSE file exists."""
        license_file = project_root / "LICENSE"
        assert license_file.exists(), "LICENSE not found"

    def test_source_directory_structure(self, project_root):
        """Test that source directory has correct structure."""
        src_dir = project_root / "src" / "banana_appeal"
        assert src_dir.exists(), "src/banana_appeal not found"

        init_file = src_dir / "__init__.py"
        assert init_file.exists(), "__init__.py not found"

        server_file = src_dir / "server.py"
        assert server_file.exists(), "server.py not found"

        models_file = src_dir / "models.py"
        assert models_file.exists(), "models.py not found"

    def test_build_package(self, project_root, tmp_path):
        """Test that the package can be built."""
        # Build the package
        result = subprocess.run(
            [sys.executable, "-m", "build", "--outdir", str(tmp_path)],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        # Check build succeeded
        assert result.returncode == 0, f"Build failed: {result.stderr}"

        # Check that wheel and sdist were created
        wheels = list(tmp_path.glob("*.whl"))
        assert len(wheels) >= 1, "No wheel file created"

        sdists = list(tmp_path.glob("*.tar.gz"))
        assert len(sdists) >= 1, "No sdist file created"


class TestEntryPoint:
    """Tests for the CLI entry point."""

    def test_entry_point_callable(self):
        """Test that the main entry point is callable."""
        from banana_appeal.server import main

        assert callable(main)

    def test_cli_import(self):
        """Test that CLI module can be imported correctly."""
        # This tests that the package installs correctly with the entry point
        result = subprocess.run(
            [sys.executable, "-c", "from banana_appeal.server import main; print('OK')"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Entry point check failed: {result.stderr}"
        assert "OK" in result.stdout
