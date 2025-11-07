"""
Benchmark import service for SPEC CPU benchmark data
"""
import csv
import io
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, desc

from models.database import Benchmark


@dataclass
class SPECImportResult:
    """Result of SPEC benchmark import"""
    total_rows: int
    imported_entries: int
    errors: List[str]
    warnings: List[str]
    benchmark_types: Dict[str, int]


@dataclass
class PassMarkImportResult:
    """Result of PassMark benchmark import"""
    total_rows: int
    imported_entries: int
    errors: List[str]
    warnings: List[str]
    benchmark_types: Dict[str, int]


@dataclass
class BenchmarkVersionInfo:
    """Information about a benchmark version"""
    version_group_id: str
    version_count: int
    latest_version_id: int
    latest_version_created_at: str
    versions: List[Dict[str, Any]]


@dataclass
class VersionManagementResult:
    """Result of version management operations"""
    operation: str
    success: bool
    message: str
    affected_versions: int
    errors: List[str]


class BenchmarkImportService:
    """Service for importing SPEC benchmark data from CSV files"""

    # SPEC benchmark types and their mappings
    SPEC_TYPES = {
        'SPECspeed Integer': {
            'category': 'cpu',
            'score_type': 'speed_integer',
            'parallel_column': 'Parallel'
        },
        'SPECspeed Floating Point': {
            'category': 'cpu',
            'score_type': 'speed_floating_point',
            'parallel_column': 'Parallel'
        },
        'SPECrate Integer': {
            'category': 'cpu',
            'score_type': 'rate_integer',
            'parallel_column': None  # Rate tests don't have parallel column
        },
        'SPECrate Floating Point': {
            'category': 'cpu',
            'score_type': 'rate_floating_point',
            'parallel_column': None  # Rate tests don't have parallel column
        }
    }

    # PassMark benchmark types and their mappings
    PASSMARK_TYPES = {
        'PASSMARK_CPU': {
            'category': 'cpu',
            'score_type': 'passmark_cpu'
        },
        'PASSMARK_GPU': {
            'category': 'gpu',
            'score_type': 'passmark_gpu'
        }
    }

    # Required columns for SPEC CSV
    REQUIRED_COLUMNS = [
        'Test Sponsor',
        'System Name',
        'Base Threads/Copies',
        'Processor - Enabled Cores',
        'Processor - Enabled Chips',
        'Processor - Threads/Core',
        'Results - Base',
        'Results - Peak'
    ]

    # Optional columns
    OPTIONAL_COLUMNS = [
        'Parallel',  # Only for Speed tests
        'Energy - Base',
        'Energy - Peak'
    ]

    # Required columns for PassMark CSV
    PASSMARK_REQUIRED_COLUMNS = [
        'Chipset Name',
        'Score'
    ]

    def __init__(self):
        self.version = "2023"  # Default SPEC version

    def _generate_content_hash(self, benchmark_data: Dict[str, Any]) -> str:
        """Generate SHA256 hash of benchmark content for duplicate detection"""
        # Create a normalized representation of the benchmark data
        content_parts = [
            benchmark_data.get('name', ''),
            benchmark_data.get('category', ''),
            benchmark_data.get('component_name', ''),
            benchmark_data.get('component_model', ''),
            benchmark_data.get('vendor', ''),
            str(benchmark_data.get('score', '')),
            benchmark_data.get('score_type', ''),
            json.dumps(benchmark_data.get('benchmark_metadata', {}), sort_keys=True)
        ]

        content_string = '|'.join(content_parts)
        return hashlib.sha256(content_string.encode('utf-8')).hexdigest()

    def _generate_version_group_id(self, benchmark_data: Dict[str, Any]) -> str:
        """Generate version group ID based on benchmark identity"""
        # Group by name, category, component_name, and score_type
        group_parts = [
            benchmark_data.get('name', ''),
            benchmark_data.get('category', ''),
            benchmark_data.get('component_name', ''),
            benchmark_data.get('score_type', '')
        ]

        group_string = '|'.join(group_parts)
        return hashlib.md5(group_string.encode('utf-8')).hexdigest()

    async def _check_duplicate_benchmark(self, content_hash: str, db: AsyncSession) -> Optional[Benchmark]:
        """Check if a benchmark with the same content hash already exists"""
        result = await db.execute(
            select(Benchmark).where(Benchmark.content_hash == content_hash)
        )
        return result.scalar_one_or_none()

    async def _get_or_create_version_group(self, benchmark_data: Dict[str, Any], db: AsyncSession) -> str:
        """Get existing version group ID or create new one"""
        version_group_id = self._generate_version_group_id(benchmark_data)

        # Check if any benchmark exists in this group
        result = await db.execute(
            select(Benchmark).where(Benchmark.version_group_id == version_group_id).limit(1)
        )
        existing = result.scalar_one_or_none()

        if existing:
            return version_group_id

        return version_group_id  # New group

    async def _update_version_flags(self, version_group_id: str, latest_benchmark_id: int, db: AsyncSession):
        """Update is_latest_version flags for all benchmarks in a group"""
        # Set all benchmarks in the group to not latest
        await db.execute(
            update(Benchmark)
            .where(Benchmark.version_group_id == version_group_id)
            .values(is_latest_version=False)
        )

        # Set the latest one to true
        await db.execute(
            update(Benchmark)
            .where(Benchmark.id == latest_benchmark_id)
            .values(is_latest_version=True)
        )

    async def _cleanup_old_versions(self, version_group_id: str, db: AsyncSession, keep_count: int = 10) -> int:
        """Remove old versions, keeping only the specified number of most recent ones"""
        # Get all versions in the group, ordered by creation date (newest first)
        result = await db.execute(
            select(Benchmark)
            .where(Benchmark.version_group_id == version_group_id)
            .order_by(desc(Benchmark.created_at))
        )
        all_versions = result.scalars().all()

        if len(all_versions) <= keep_count:
            return 0  # No cleanup needed

        # Keep the most recent 'keep_count' versions
        versions_to_keep = all_versions[:keep_count]
        versions_to_delete = all_versions[keep_count:]

        # Delete old versions
        deleted_count = 0
        for old_version in versions_to_delete:
            await db.delete(old_version)
            deleted_count += 1

        logger.info(f"Cleaned up {deleted_count} old versions for group {version_group_id}")
        return deleted_count

    async def import_spec_csv(
        self,
        csv_content: bytes,
        filename: str,
        db: AsyncSession,
        version: Optional[str] = None
    ) -> SPECImportResult:
        """
        Import SPEC benchmark data from CSV file

        Args:
            csv_content: CSV file content as bytes
            filename: Original filename
            db: Database session
            version: SPEC version (optional)

        Returns:
            SPECImportResult with import statistics and errors
        """
        if version:
            self.version = version

        result = SPECImportResult(
            total_rows=0,
            imported_entries=0,
            errors=[],
            warnings=[],
            benchmark_types={}
        )

        try:
            # Parse CSV content
            rows = self._parse_csv(csv_content, filename)
            result.total_rows = len(rows)

            if not rows:
                result.errors.append("No data rows found in CSV file")
                return result

            # Validate CSV structure
            validation_result = self._validate_csv_structure(rows[0])
            if not validation_result['valid']:
                result.errors.extend(validation_result['errors'])
                return result

            # Process each row
            for row_idx, row in enumerate(rows, 1):
                try:
                    benchmark_entries = self._process_spec_row(row, row_idx)
                    if benchmark_entries:
                        # Save to database
                        saved_count = await self._save_benchmark_entries(benchmark_entries, db)
                        result.imported_entries += saved_count

                        # Update benchmark types count
                        for entry in benchmark_entries:
                            bench_type = entry['score_type']
                            result.benchmark_types[bench_type] = result.benchmark_types.get(bench_type, 0) + 1

                except Exception as e:
                    result.errors.append(f"Row {row_idx}: {str(e)}")
                    logger.error(f"Error processing row {row_idx}: {e}")

        except Exception as e:
            result.errors.append(f"Failed to process CSV file: {str(e)}")
            logger.error(f"Error importing SPEC CSV {filename}: {e}")

        return result

    async def import_passmark_csv(
        self,
        csv_content: bytes,
        filename: str,
        db: AsyncSession,
        benchmark_type: str = "PASSMARK_CPU"  # PASSMARK_CPU or PASSMARK_GPU
    ) -> PassMarkImportResult:
        """
        Import PassMark benchmark data from CSV file

        Args:
            csv_content: CSV file content as bytes
            filename: Original filename
            db: Database session
            benchmark_type: Type of PassMark benchmark (PASSMARK_CPU or PASSMARK_GPU)

        Returns:
            PassMarkImportResult with import statistics and errors
        """
        if benchmark_type not in self.PASSMARK_TYPES:
            raise ValueError(f"Invalid benchmark type: {benchmark_type}. Must be PASSMARK_CPU or PASSMARK_GPU")

        result = PassMarkImportResult(
            total_rows=0,
            imported_entries=0,
            errors=[],
            warnings=[],
            benchmark_types={}
        )

        try:
            # Parse CSV content
            rows = self._parse_csv(csv_content, filename)
            result.total_rows = len(rows)

            if not rows:
                result.errors.append("No data rows found in CSV file")
                return result

            # Validate CSV structure
            validation_result = self._validate_passmark_csv_structure(rows[0])
            if not validation_result['valid']:
                result.errors.extend(validation_result['errors'])
                return result

            # Process each row
            for row_idx, row in enumerate(rows, 1):
                try:
                    benchmark_entries = self._process_passmark_row(row, row_idx, benchmark_type)
                    if benchmark_entries:
                        # Save to database
                        saved_count = await self._save_benchmark_entries(benchmark_entries, db)
                        result.imported_entries += saved_count

                        # Update benchmark types count
                        for entry in benchmark_entries:
                            bench_type = entry['score_type']
                            result.benchmark_types[bench_type] = result.benchmark_types.get(bench_type, 0) + 1

                except Exception as e:
                    result.errors.append(f"Row {row_idx}: {str(e)}")
                    logger.error(f"Error processing row {row_idx}: {e}")

        except Exception as e:
            result.errors.append(f"Failed to process CSV file: {str(e)}")
            logger.error(f"Error importing PassMark CSV {filename}: {e}")

        return result

    def _parse_csv(self, csv_content: bytes, filename: str) -> List[Dict[str, str]]:
        """Parse CSV content into list of dictionaries"""
        try:
            # Try to decode with different encodings
            text_content = None
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    text_content = csv_content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if text_content is None:
                raise ValueError("Unable to decode CSV file with supported encodings")

            # Parse CSV
            csv_reader = csv.DictReader(io.StringIO(text_content))
            rows = list(csv_reader)

            logger.info(f"Parsed {len(rows)} rows from CSV file {filename}")
            return rows

        except Exception as e:
            logger.error(f"Error parsing CSV {filename}: {e}")
            raise

    def _validate_csv_structure(self, first_row: Dict[str, str]) -> Dict[str, Any]:
        """Validate that CSV has required columns"""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        columns = list(first_row.keys())

        # Check required columns
        missing_required = []
        for req_col in self.REQUIRED_COLUMNS:
            if req_col not in columns:
                missing_required.append(req_col)

        if missing_required:
            validation['valid'] = False
            validation['errors'].append(f"Missing required columns: {', '.join(missing_required)}")

        # Check for Parallel column (important for Speed tests)
        has_parallel = 'Parallel' in columns
        if not has_parallel:
            validation['warnings'].append("Parallel column not found - Speed tests may not be properly identified")

        # Check energy columns
        has_energy = 'Energy - Base' in columns and 'Energy - Peak' in columns
        if not has_energy:
            validation['warnings'].append("Energy columns not found - energy data will not be imported")

        return validation

    def _validate_passmark_csv_structure(self, first_row: Dict[str, str]) -> Dict[str, Any]:
        """Validate that PassMark CSV has required columns"""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        columns = list(first_row.keys())

        # Check required columns
        missing_required = []
        for req_col in self.PASSMARK_REQUIRED_COLUMNS:
            if req_col not in columns:
                missing_required.append(req_col)

        if missing_required:
            validation['valid'] = False
            validation['errors'].append(f"Missing required columns: {', '.join(missing_required)}")

        return validation

    def _process_spec_row(self, row: Dict[str, str], row_idx: int) -> List[Dict[str, Any]]:
        """
        Process a single SPEC CSV row into benchmark entries

        Returns list of benchmark entry dictionaries ready for database insertion
        """
        entries = []

        try:
            # Extract common data
            test_sponsor = row.get('Test Sponsor', '').strip()
            system_name = row.get('System Name', '').strip()
            base_threads = self._parse_int(row.get('Base Threads/Copies', ''))
            processor_cores = self._parse_int(row.get('Processor - Enabled Cores', ''))
            processor_chips = self._parse_int(row.get('Processor - Enabled Chips', ''))
            threads_per_core = self._parse_int(row.get('Processor - Threads/Core', ''))

            # Results
            results_base = self._parse_float(row.get('Results - Base', ''))
            results_peak = self._parse_float(row.get('Results - Peak', ''))

            # Energy (optional)
            energy_base = self._parse_float(row.get('Energy - Base', ''))
            energy_peak = self._parse_float(row.get('Energy - Peak', ''))

            # Parallel (only for Speed tests)
            parallel = row.get('Parallel', '').strip()

            # Determine benchmark types based on available data
            benchmark_types = self._determine_benchmark_types(row, parallel)

            for bench_type, config in benchmark_types.items():
                # Create base entry
                entry = {
                    'name': f'SPEC {bench_type}',
                    'category': config['category'],
                    'version': self.version,
                    'component_name': system_name,
                    'component_model': f"{processor_cores}C/{threads_per_core}T x {processor_chips}",
                    'vendor': test_sponsor,
                    'score_type': config['score_type'],
                    'benchmark_metadata': {
                        'base_threads': base_threads,
                        'processor_cores': processor_cores,
                        'processor_chips': processor_chips,
                        'threads_per_core': threads_per_core,
                        'parallel': parallel if parallel else None,
                        'energy_base': energy_base,
                        'energy_peak': energy_peak,
                        'source_row': row_idx
                    }
                }

                # Add base score if available
                if results_base is not None:
                    base_entry = entry.copy()
                    base_entry['score'] = results_base
                    base_entry['benchmark_metadata'] = entry['benchmark_metadata'].copy()
                    base_entry['benchmark_metadata']['result_type'] = 'base'
                    entries.append(base_entry)

                # Add peak score if available
                if results_peak is not None:
                    peak_entry = entry.copy()
                    peak_entry['score'] = results_peak
                    peak_entry['benchmark_metadata'] = entry['benchmark_metadata'].copy()
                    peak_entry['benchmark_metadata']['result_type'] = 'peak'
                    entries.append(peak_entry)

        except Exception as e:
            logger.error(f"Error processing row {row_idx}: {e}")
            raise

        return entries

    def _process_passmark_row(self, row: Dict[str, str], row_idx: int, benchmark_type: str) -> List[Dict[str, Any]]:
        """
        Process a single PassMark CSV row into benchmark entries

        Returns list of benchmark entry dictionaries ready for database insertion
        """
        entries = []

        try:
            # Extract data
            chipset_name = row.get('Chipset Name', '').strip()
            score = self._parse_float(row.get('Score', ''))

            if not chipset_name:
                raise ValueError("Chipset Name is required")

            if score is None:
                raise ValueError("Score is required and must be a valid number")

            # Get benchmark configuration
            config = self.PASSMARK_TYPES[benchmark_type]

            # Create benchmark entry
            entry = {
                'name': f'PassMark {benchmark_type}',
                'category': config['category'],
                'version': 'latest',  # PassMark doesn't have versions like SPEC
                'component_name': chipset_name,
                'component_model': chipset_name,  # For PassMark, chipset name is the model
                'vendor': '',  # PassMark doesn't specify vendor in CSV
                'score_type': config['score_type'],
                'score': score,
                'benchmark_metadata': {
                    'source_row': row_idx,
                    'benchmark_type': benchmark_type
                }
            }

            entries.append(entry)

        except Exception as e:
            logger.error(f"Error processing PassMark row {row_idx}: {e}")
            raise

        return entries

    def _determine_benchmark_types(self, row: Dict[str, str], parallel: str) -> Dict[str, Dict[str, str]]:
        """Determine which benchmark types this row represents"""
        types = {}

        # Check for Speed tests (have Parallel column)
        if parallel:
            if 'SPECspeed Integer' in str(row) or self._is_speed_integer_test(row):
                types['SPECspeed Integer'] = self.SPEC_TYPES['SPECspeed Integer']
            if 'SPECspeed Floating Point' in str(row) or self._is_speed_floating_test(row):
                types['SPECspeed Floating Point'] = self.SPEC_TYPES['SPECspeed Floating Point']

        # Check for Rate tests (no Parallel column or Parallel is empty)
        if not parallel or not parallel.strip():
            if 'SPECrate Integer' in str(row) or self._is_rate_integer_test(row):
                types['SPECrate Integer'] = self.SPEC_TYPES['SPECrate Integer']
            if 'SPECrate Floating Point' in str(row) or self._is_rate_floating_test(row):
                types['SPECrate Floating Point'] = self.SPEC_TYPES['SPECrate Floating Point']

        # If no specific types detected, try to infer from column names or data
        if not types:
            # Look for specific patterns in the row data
            row_str = str(row).upper()
            if 'SPEED' in row_str and 'INTEGER' in row_str:
                types['SPECspeed Integer'] = self.SPEC_TYPES['SPECspeed Integer']
            elif 'SPEED' in row_str and 'FLOAT' in row_str:
                types['SPECspeed Floating Point'] = self.SPEC_TYPES['SPECspeed Floating Point']
            elif 'RATE' in row_str and 'INTEGER' in row_str:
                types['SPECrate Integer'] = self.SPEC_TYPES['SPECrate Integer']
            elif 'RATE' in row_str and 'FLOAT' in row_str:
                types['SPECrate Floating Point'] = self.SPEC_TYPES['SPECrate Floating Point']

        return types

    def _is_speed_integer_test(self, row: Dict[str, str]) -> bool:
        """Check if row represents SPECspeed Integer test"""
        # This is a heuristic - in real SPEC CSV, this would be more sophisticated
        return True  # Placeholder - would need actual SPEC CSV analysis

    def _is_speed_floating_test(self, row: Dict[str, str]) -> bool:
        """Check if row represents SPECspeed Floating Point test"""
        return True  # Placeholder

    def _is_rate_integer_test(self, row: Dict[str, str]) -> bool:
        """Check if row represents SPECrate Integer test"""
        return True  # Placeholder

    def _is_rate_floating_test(self, row: Dict[str, str]) -> bool:
        """Check if row represents SPECrate Floating Point test"""
        return True  # Placeholder

    def _parse_int(self, value: str) -> Optional[int]:
        """Parse string to int, returning None if invalid"""
        if not value or not value.strip():
            return None
        try:
            return int(float(value.strip()))
        except (ValueError, TypeError):
            return None

    def _parse_float(self, value: str) -> Optional[float]:
        """Parse string to float, returning None if invalid"""
        if not value or not value.strip():
            return None
        try:
            return float(value.strip())
        except (ValueError, TypeError):
            return None

    async def _save_benchmark_entries(self, entries: List[Dict[str, Any]], db: AsyncSession) -> int:
        """Save benchmark entries to database with version management"""
        saved_count = 0

        for entry in entries:
            try:
                # Generate content hash and version group ID
                content_hash = self._generate_content_hash(entry)
                version_group_id = await self._get_or_create_version_group(entry, db)

                # Check for exact duplicate
                duplicate = await self._check_duplicate_benchmark(content_hash, db)
                if duplicate:
                    logger.info(f"Skipping duplicate benchmark: {entry.get('name')} - {entry.get('component_name')}")
                    continue

                # Add version management fields to entry
                entry['content_hash'] = content_hash
                entry['version_group_id'] = version_group_id
                entry['is_latest_version'] = True  # New entries are initially the latest

                # Create new benchmark version
                benchmark = Benchmark(**entry)
                db.add(benchmark)
                await db.flush()  # Get the ID

                # Update version flags for the group
                await self._update_version_flags(version_group_id, benchmark.id, db)

                # Cleanup old versions (keep last 10)
                await self._cleanup_old_versions(version_group_id, db, keep_count=10)

                saved_count += 1

            except Exception as e:
                logger.error(f"Error saving benchmark entry: {e}")
                continue

        await db.commit()
        return saved_count

    async def _find_existing_benchmark(self, entry: Dict[str, Any], db: AsyncSession) -> Optional[Benchmark]:
        """Find existing benchmark with same characteristics (legacy method for compatibility)"""
        try:
            # For PassMark benchmarks, use different matching criteria since they don't have result_type
            if entry['score_type'] in ['passmark_cpu', 'passmark_gpu']:
                result = await db.execute(
                    select(Benchmark).where(
                        Benchmark.name == entry['name'],
                        Benchmark.category == entry['category'],
                        Benchmark.component_name == entry['component_name'],
                        Benchmark.score_type == entry['score_type']
                    )
                )
            else:
                # For SPEC benchmarks, use the original logic
                result = await db.execute(
                    select(Benchmark).where(
                        Benchmark.name == entry['name'],
                        Benchmark.category == entry['category'],
                        Benchmark.component_name == entry['component_name'],
                        Benchmark.score_type == entry['score_type'],
                        Benchmark.benchmark_metadata['result_type'].astext == entry['benchmark_metadata']['result_type']
                    )
                )
            return result.scalar_one_or_none()
        except Exception:
            return None

    # Version Management Methods

    async def create_benchmark_version(self, benchmark_data: Dict[str, Any], db: AsyncSession) -> Benchmark:
        """
        Create a new version of a benchmark

        Args:
            benchmark_data: Benchmark data dictionary
            db: Database session

        Returns:
            Created Benchmark instance
        """
        # Generate content hash and version group ID
        content_hash = self._generate_content_hash(benchmark_data)
        version_group_id = await self._get_or_create_version_group(benchmark_data, db)

        # Check for exact duplicate
        duplicate = await self._check_duplicate_benchmark(content_hash, db)
        if duplicate:
            raise ValueError(f"Benchmark with identical content already exists (ID: {duplicate.id})")

        # Add version management fields
        benchmark_data['content_hash'] = content_hash
        benchmark_data['version_group_id'] = version_group_id
        benchmark_data['is_latest_version'] = True

        # Create new benchmark version
        benchmark = Benchmark(**benchmark_data)
        db.add(benchmark)
        await db.flush()

        # Update version flags for the group
        await self._update_version_flags(version_group_id, benchmark.id, db)

        # Cleanup old versions
        await self._cleanup_old_versions(version_group_id, db, keep_count=10)

        await db.commit()
        return benchmark

    async def get_benchmark_versions(self, version_group_id: str, db: AsyncSession) -> BenchmarkVersionInfo:
        """
        Get all versions of a benchmark

        Args:
            version_group_id: Version group identifier
            db: Database session

        Returns:
            BenchmarkVersionInfo with version details
        """
        result = await db.execute(
            select(Benchmark)
            .where(Benchmark.version_group_id == version_group_id)
            .order_by(desc(Benchmark.created_at))
        )
        versions = result.scalars().all()

        if not versions:
            raise ValueError(f"No benchmarks found for version group: {version_group_id}")

        latest_version = versions[0]  # Already ordered by creation date

        version_list = []
        for v in versions:
            version_list.append({
                'id': v.id,
                'version': v.version,
                'score': v.score,
                'is_latest_version': v.is_latest_version,
                'created_at': v.created_at.isoformat() if v.created_at else None,
                'content_hash': v.content_hash
            })

        return BenchmarkVersionInfo(
            version_group_id=version_group_id,
            version_count=len(versions),
            latest_version_id=latest_version.id,
            latest_version_created_at=latest_version.created_at.isoformat() if latest_version.created_at else None,
            versions=version_list
        )

    async def get_benchmark_version_groups(self, db: AsyncSession, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get list of all benchmark version groups with their latest versions

        Args:
            db: Database session
            limit: Maximum number of groups to return
            offset: Offset for pagination

        Returns:
            List of version group summaries
        """
        # Get distinct version groups with their latest versions
        result = await db.execute(
            select(
                Benchmark.version_group_id,
                Benchmark.name,
                Benchmark.category,
                Benchmark.component_name,
                Benchmark.score_type,
                func.count(Benchmark.id).label('version_count'),
                func.max(Benchmark.created_at).label('latest_created_at')
            )
            .where(Benchmark.is_latest_version == True)
            .group_by(
                Benchmark.version_group_id,
                Benchmark.name,
                Benchmark.category,
                Benchmark.component_name,
                Benchmark.score_type
            )
            .order_by(desc(func.max(Benchmark.created_at)))
            .limit(limit)
            .offset(offset)
        )

        groups = []
        for row in result:
            groups.append({
                'version_group_id': row.version_group_id,
                'name': row.name,
                'category': row.category,
                'component_name': row.component_name,
                'score_type': row.score_type,
                'version_count': row.version_count,
                'latest_created_at': row.latest_created_at.isoformat() if row.latest_created_at else None
            })

        return groups

    async def cleanup_old_versions(self, db: AsyncSession, max_versions_per_group: int = 10) -> VersionManagementResult:
        """
        Cleanup old versions across all benchmark groups

        Args:
            max_versions_per_group: Maximum versions to keep per group
            db: Database session

        Returns:
            VersionManagementResult with cleanup statistics
        """
        errors = []
        total_deleted = 0

        try:
            # Get all version groups
            result = await db.execute(
                select(Benchmark.version_group_id).distinct()
            )
            version_groups = [row[0] for row in result]

            for group_id in version_groups:
                try:
                    deleted = await self._cleanup_old_versions(group_id, db, keep_count=max_versions_per_group)
                    total_deleted += deleted
                except Exception as e:
                    errors.append(f"Error cleaning up group {group_id}: {str(e)}")

            await db.commit()

            return VersionManagementResult(
                operation="cleanup_old_versions",
                success=True,
                message=f"Cleaned up {total_deleted} old benchmark versions",
                affected_versions=total_deleted,
                errors=errors
            )

        except Exception as e:
            await db.rollback()
            return VersionManagementResult(
                operation="cleanup_old_versions",
                success=False,
                message=f"Cleanup failed: {str(e)}",
                affected_versions=0,
                errors=[str(e)]
            )

    async def set_active_version(self, benchmark_id: int, db: AsyncSession) -> VersionManagementResult:
        """
        Set a specific benchmark version as the active (latest) version

        Args:
            benchmark_id: ID of the benchmark to set as active
            db: Database session

        Returns:
            VersionManagementResult with operation result
        """
        try:
            # Get the benchmark
            result = await db.execute(
                select(Benchmark).where(Benchmark.id == benchmark_id)
            )
            benchmark = result.scalar_one_or_none()

            if not benchmark:
                return VersionManagementResult(
                    operation="set_active_version",
                    success=False,
                    message=f"Benchmark with ID {benchmark_id} not found",
                    affected_versions=0,
                    errors=["Benchmark not found"]
                )

            # Update version flags for the group
            await self._update_version_flags(benchmark.version_group_id, benchmark_id, db)
            await db.commit()

            return VersionManagementResult(
                operation="set_active_version",
                success=True,
                message=f"Set benchmark {benchmark_id} as active version",
                affected_versions=1,
                errors=[]
            )

        except Exception as e:
            await db.rollback()
            return VersionManagementResult(
                operation="set_active_version",
                success=False,
                message=f"Failed to set active version: {str(e)}",
                affected_versions=0,
                errors=[str(e)]
            )

    async def delete_benchmark_version(self, benchmark_id: int, db: AsyncSession) -> VersionManagementResult:
        """
        Delete a specific benchmark version

        Args:
            benchmark_id: ID of the benchmark version to delete
            db: Database session

        Returns:
            VersionManagementResult with operation result
        """
        try:
            # Get the benchmark
            result = await db.execute(
                select(Benchmark).where(Benchmark.id == benchmark_id)
            )
            benchmark = result.scalar_one_or_none()

            if not benchmark:
                return VersionManagementResult(
                    operation="delete_benchmark_version",
                    success=False,
                    message=f"Benchmark with ID {benchmark_id} not found",
                    affected_versions=0,
                    errors=["Benchmark not found"]
                )

            version_group_id = benchmark.version_group_id
            was_latest = benchmark.is_latest_version

            # Delete the benchmark
            await db.delete(benchmark)

            # If this was the latest version, set the next most recent as latest
            if was_latest:
                result = await db.execute(
                    select(Benchmark)
                    .where(Benchmark.version_group_id == version_group_id)
                    .order_by(desc(Benchmark.created_at))
                    .limit(1)
                )
                next_latest = result.scalar_one_or_none()
                if next_latest:
                    next_latest.is_latest_version = True

            await db.commit()

            return VersionManagementResult(
                operation="delete_benchmark_version",
                success=True,
                message=f"Deleted benchmark version {benchmark_id}",
                affected_versions=1,
                errors=[]
            )

        except Exception as e:
            await db.rollback()
            return VersionManagementResult(
                operation="delete_benchmark_version",
                success=False,
                message=f"Failed to delete benchmark version: {str(e)}",
                affected_versions=0,
                errors=[str(e)]
            )


# Singleton instance
benchmark_import_service = BenchmarkImportService()