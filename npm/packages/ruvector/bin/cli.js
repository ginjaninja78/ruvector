#!/usr/bin/env node

const { Command } = require('commander');
const chalk = require('chalk');
const ora = require('ora');
const fs = require('fs');
const path = require('path');

// Lazy load ruvector (only when needed, not for install/help commands)
let VectorDB, getVersion, getImplementationType;
let ruvectorLoaded = false;

function loadRuvector() {
  if (ruvectorLoaded) return true;
  try {
    const ruvector = require('../dist/index.js');
    VectorDB = ruvector.VectorDB;
    getVersion = ruvector.getVersion;
    getImplementationType = ruvector.getImplementationType;
    ruvectorLoaded = true;
    return true;
  } catch (e) {
    return false;
  }
}

function requireRuvector() {
  if (!loadRuvector()) {
    console.error(chalk.red('Error: Failed to load ruvector. Please run: npm run build'));
    console.error(chalk.yellow('Or install the package: npm install ruvector'));
    process.exit(1);
  }
}

// Import GNN (optional - graceful fallback if not available)
let RuvectorLayer, TensorCompress, differentiableSearch, getCompressionLevel, hierarchicalForward;
let gnnAvailable = false;
try {
  const gnn = require('@ruvector/gnn');
  RuvectorLayer = gnn.RuvectorLayer;
  TensorCompress = gnn.TensorCompress;
  differentiableSearch = gnn.differentiableSearch;
  getCompressionLevel = gnn.getCompressionLevel;
  hierarchicalForward = gnn.hierarchicalForward;
  gnnAvailable = true;
} catch (e) {
  // GNN not available - commands will show helpful message
}

const program = new Command();

// Get package version from package.json
const packageJson = require('../package.json');

// Version and description (lazy load implementation info)
program
  .name('ruvector')
  .description(`${chalk.cyan('ruvector')} - High-performance vector database CLI`)
  .version(packageJson.version);

// Create database
program
  .command('create <path>')
  .description('Create a new vector database')
  .option('-d, --dimension <number>', 'Vector dimension', '384')
  .option('-m, --metric <type>', 'Distance metric (cosine|euclidean|dot)', 'cosine')
  .action((dbPath, options) => {
    requireRuvector();
    const spinner = ora('Creating database...').start();

    try {
      const dimension = parseInt(options.dimension);
      const db = new VectorDB({
        dimension,
        metric: options.metric,
        path: dbPath,
        autoPersist: true
      });

      db.save(dbPath);
      spinner.succeed(chalk.green(`Database created: ${dbPath}`));
      console.log(chalk.gray(`  Dimension: ${dimension}`));
      console.log(chalk.gray(`  Metric: ${options.metric}`));
      console.log(chalk.gray(`  Implementation: ${getImplementationType()}`));
    } catch (error) {
      spinner.fail(chalk.red('Failed to create database'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// Insert vectors
program
  .command('insert <database> <file>')
  .description('Insert vectors from JSON file')
  .option('-b, --batch-size <number>', 'Batch size for insertion', '1000')
  .action((dbPath, file, options) => {
    requireRuvector();
    const spinner = ora('Loading database...').start();

    try {
      // Read database metadata to get dimension
      let dimension = 384; // default
      if (fs.existsSync(dbPath)) {
        const dbData = fs.readFileSync(dbPath, 'utf8');
        const parsed = JSON.parse(dbData);
        dimension = parsed.dimension || 384;
      }

      const db = new VectorDB({ dimension });

      if (fs.existsSync(dbPath)) {
        db.load(dbPath);
      }

      spinner.text = 'Reading vectors...';
      const data = JSON.parse(fs.readFileSync(file, 'utf8'));
      const vectors = Array.isArray(data) ? data : [data];

      spinner.text = `Inserting ${vectors.length} vectors...`;
      const batchSize = parseInt(options.batchSize);

      for (let i = 0; i < vectors.length; i += batchSize) {
        const batch = vectors.slice(i, i + batchSize);
        db.insertBatch(batch);
        spinner.text = `Inserted ${Math.min(i + batchSize, vectors.length)}/${vectors.length} vectors...`;
      }

      db.save(dbPath);
      spinner.succeed(chalk.green(`Inserted ${vectors.length} vectors`));

      const stats = db.stats();
      console.log(chalk.gray(`  Total vectors: ${stats.count}`));
    } catch (error) {
      spinner.fail(chalk.red('Failed to insert vectors'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// Search vectors
program
  .command('search <database>')
  .description('Search for similar vectors')
  .requiredOption('-v, --vector <json>', 'Query vector as JSON array')
  .option('-k, --top-k <number>', 'Number of results', '10')
  .option('-t, --threshold <number>', 'Similarity threshold', '0.0')
  .option('-f, --filter <json>', 'Metadata filter as JSON')
  .action((dbPath, options) => {
    requireRuvector();
    const spinner = ora('Loading database...').start();

    try {
      // Read database metadata
      const dbData = fs.readFileSync(dbPath, 'utf8');
      const parsed = JSON.parse(dbData);
      const dimension = parsed.dimension || 384;

      const db = new VectorDB({ dimension });
      db.load(dbPath);

      spinner.text = 'Searching...';

      const vector = JSON.parse(options.vector);
      const query = {
        vector,
        k: parseInt(options.topK),
        threshold: parseFloat(options.threshold)
      };

      if (options.filter) {
        query.filter = JSON.parse(options.filter);
      }

      const results = db.search(query);
      spinner.succeed(chalk.green(`Found ${results.length} results`));

      console.log(chalk.cyan('\nSearch Results:'));
      results.forEach((result, i) => {
        console.log(chalk.white(`\n${i + 1}. ID: ${result.id}`));
        console.log(chalk.yellow(`   Score: ${result.score.toFixed(4)}`));
        if (result.metadata) {
          console.log(chalk.gray(`   Metadata: ${JSON.stringify(result.metadata)}`));
        }
      });
    } catch (error) {
      spinner.fail(chalk.red('Failed to search'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// Show stats
program
  .command('stats <database>')
  .description('Show database statistics')
  .action((dbPath) => {
    requireRuvector();
    const spinner = ora('Loading database...').start();

    try {
      const dbData = fs.readFileSync(dbPath, 'utf8');
      const parsed = JSON.parse(dbData);
      const dimension = parsed.dimension || 384;

      const db = new VectorDB({ dimension });
      db.load(dbPath);

      const stats = db.stats();
      spinner.succeed(chalk.green('Database statistics'));

      console.log(chalk.cyan('\nDatabase Stats:'));
      console.log(chalk.white(`  Vector Count: ${chalk.yellow(stats.count)}`));
      console.log(chalk.white(`  Dimension: ${chalk.yellow(stats.dimension)}`));
      console.log(chalk.white(`  Metric: ${chalk.yellow(stats.metric)}`));
      console.log(chalk.white(`  Implementation: ${chalk.yellow(getImplementationType())}`));

      if (stats.memoryUsage) {
        const mb = (stats.memoryUsage / (1024 * 1024)).toFixed(2);
        console.log(chalk.white(`  Memory Usage: ${chalk.yellow(mb + ' MB')}`));
      }

      const fileStats = fs.statSync(dbPath);
      const fileMb = (fileStats.size / (1024 * 1024)).toFixed(2);
      console.log(chalk.white(`  File Size: ${chalk.yellow(fileMb + ' MB')}`));
    } catch (error) {
      spinner.fail(chalk.red('Failed to load database'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// Benchmark
program
  .command('benchmark')
  .description('Run performance benchmarks')
  .option('-d, --dimension <number>', 'Vector dimension', '384')
  .option('-n, --num-vectors <number>', 'Number of vectors', '10000')
  .option('-q, --num-queries <number>', 'Number of queries', '1000')
  .action((options) => {
    requireRuvector();
    console.log(chalk.cyan('\nruvector Performance Benchmark'));
    console.log(chalk.gray(`Implementation: ${getImplementationType()}\n`));

    const dimension = parseInt(options.dimension);
    const numVectors = parseInt(options.numVectors);
    const numQueries = parseInt(options.numQueries);

    let spinner = ora('Creating database...').start();

    try {
      const db = new VectorDB({ dimension, metric: 'cosine' });
      spinner.succeed();

      // Insert benchmark
      spinner = ora(`Inserting ${numVectors} vectors...`).start();
      const insertStart = Date.now();

      const vectors = [];
      for (let i = 0; i < numVectors; i++) {
        vectors.push({
          id: `vec_${i}`,
          vector: Array.from({ length: dimension }, () => Math.random()),
          metadata: { index: i, batch: Math.floor(i / 1000) }
        });
      }

      db.insertBatch(vectors);
      const insertTime = Date.now() - insertStart;
      const insertRate = (numVectors / (insertTime / 1000)).toFixed(0);

      spinner.succeed(chalk.green(`Inserted ${numVectors} vectors in ${insertTime}ms`));
      console.log(chalk.gray(`  Rate: ${chalk.yellow(insertRate)} vectors/sec`));

      // Search benchmark
      spinner = ora(`Running ${numQueries} searches...`).start();
      const searchStart = Date.now();

      for (let i = 0; i < numQueries; i++) {
        const query = {
          vector: Array.from({ length: dimension }, () => Math.random()),
          k: 10
        };
        db.search(query);
      }

      const searchTime = Date.now() - searchStart;
      const searchRate = (numQueries / (searchTime / 1000)).toFixed(0);
      const avgLatency = (searchTime / numQueries).toFixed(2);

      spinner.succeed(chalk.green(`Completed ${numQueries} searches in ${searchTime}ms`));
      console.log(chalk.gray(`  Rate: ${chalk.yellow(searchRate)} queries/sec`));
      console.log(chalk.gray(`  Avg Latency: ${chalk.yellow(avgLatency)}ms`));

      // Stats
      const stats = db.stats();
      console.log(chalk.cyan('\nFinal Stats:'));
      console.log(chalk.white(`  Vector Count: ${chalk.yellow(stats.count)}`));
      console.log(chalk.white(`  Dimension: ${chalk.yellow(stats.dimension)}`));
      console.log(chalk.white(`  Implementation: ${chalk.yellow(getImplementationType())}`));

    } catch (error) {
      spinner.fail(chalk.red('Benchmark failed'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// Info command
program
  .command('info')
  .description('Show ruvector information')
  .action(() => {
    console.log(chalk.cyan('\nruvector Information'));
    console.log(chalk.white(`  CLI Version: ${chalk.yellow(packageJson.version)}`));

    // Try to load ruvector for implementation info
    if (loadRuvector()) {
      const info = getVersion();
      console.log(chalk.white(`  Core Version: ${chalk.yellow(info.version)}`));
      console.log(chalk.white(`  Implementation: ${chalk.yellow(info.implementation)}`));
    } else {
      console.log(chalk.white(`  Core: ${chalk.gray('Not loaded (install @ruvector/core)')}`));
    }

    console.log(chalk.white(`  GNN Module: ${gnnAvailable ? chalk.green('Available') : chalk.gray('Not installed')}`));
    console.log(chalk.white(`  Node Version: ${chalk.yellow(process.version)}`));
    console.log(chalk.white(`  Platform: ${chalk.yellow(process.platform)}`));
    console.log(chalk.white(`  Architecture: ${chalk.yellow(process.arch)}`));

    if (!gnnAvailable) {
      console.log(chalk.gray('\n  Install GNN with: npx ruvector install gnn'));
    }
  });

// =============================================================================
// Install Command
// =============================================================================

program
  .command('install [packages...]')
  .description('Install optional ruvector packages')
  .option('-a, --all', 'Install all optional packages')
  .option('-l, --list', 'List available packages')
  .option('-i, --interactive', 'Interactive package selection')
  .action(async (packages, options) => {
    const { execSync } = require('child_process');

    // Available optional packages - all ruvector npm packages
    const availablePackages = {
      // Core packages
      core: {
        name: '@ruvector/core',
        description: 'Core vector database with native Rust bindings (HNSW, SIMD)',
        installed: true, // Always installed with ruvector
        category: 'core'
      },
      gnn: {
        name: '@ruvector/gnn',
        description: 'Graph Neural Network layers, tensor compression, differentiable search',
        installed: gnnAvailable,
        category: 'core'
      },
      'graph-node': {
        name: '@ruvector/graph-node',
        description: 'Native Node.js bindings for hypergraph database with Cypher queries',
        installed: false,
        category: 'core'
      },
      'agentic-synth': {
        name: '@ruvector/agentic-synth',
        description: 'Synthetic data generator for AI/ML training, RAG, and agentic workflows',
        installed: false,
        category: 'tools'
      },
      extensions: {
        name: 'ruvector-extensions',
        description: 'Advanced features: embeddings, UI, exports, temporal tracking, persistence',
        installed: false,
        category: 'tools'
      },
      // Platform-specific native bindings for @ruvector/core
      'node-linux-x64': {
        name: '@ruvector/node-linux-x64-gnu',
        description: 'Linux x64 native bindings for @ruvector/core',
        installed: false,
        category: 'platform'
      },
      'node-linux-arm64': {
        name: '@ruvector/node-linux-arm64-gnu',
        description: 'Linux ARM64 native bindings for @ruvector/core',
        installed: false,
        category: 'platform'
      },
      'node-darwin-x64': {
        name: '@ruvector/node-darwin-x64',
        description: 'macOS Intel x64 native bindings for @ruvector/core',
        installed: false,
        category: 'platform'
      },
      'node-darwin-arm64': {
        name: '@ruvector/node-darwin-arm64',
        description: 'macOS Apple Silicon native bindings for @ruvector/core',
        installed: false,
        category: 'platform'
      },
      'node-win32-x64': {
        name: '@ruvector/node-win32-x64-msvc',
        description: 'Windows x64 native bindings for @ruvector/core',
        installed: false,
        category: 'platform'
      },
      // Platform-specific native bindings for @ruvector/gnn
      'gnn-linux-x64': {
        name: '@ruvector/gnn-linux-x64-gnu',
        description: 'Linux x64 native bindings for @ruvector/gnn',
        installed: false,
        category: 'platform'
      },
      'gnn-linux-arm64': {
        name: '@ruvector/gnn-linux-arm64-gnu',
        description: 'Linux ARM64 native bindings for @ruvector/gnn',
        installed: false,
        category: 'platform'
      },
      'gnn-darwin-x64': {
        name: '@ruvector/gnn-darwin-x64',
        description: 'macOS Intel x64 native bindings for @ruvector/gnn',
        installed: false,
        category: 'platform'
      },
      'gnn-darwin-arm64': {
        name: '@ruvector/gnn-darwin-arm64',
        description: 'macOS Apple Silicon native bindings for @ruvector/gnn',
        installed: false,
        category: 'platform'
      },
      'gnn-win32-x64': {
        name: '@ruvector/gnn-win32-x64-msvc',
        description: 'Windows x64 native bindings for @ruvector/gnn',
        installed: false,
        category: 'platform'
      },
      // Legacy/standalone packages
      'ruvector-core': {
        name: 'ruvector-core',
        description: 'Standalone vector database (legacy, use @ruvector/core instead)',
        installed: false,
        category: 'legacy'
      }
    };

    // Check which packages are actually installed
    for (const [key, pkg] of Object.entries(availablePackages)) {
      if (key !== 'core' && key !== 'gnn') {
        try {
          require.resolve(pkg.name);
          pkg.installed = true;
        } catch (e) {
          pkg.installed = false;
        }
      }
    }

    // List packages
    if (options.list || (packages.length === 0 && !options.all && !options.interactive)) {
      console.log(chalk.cyan('\n═══════════════════════════════════════════════════════════════'));
      console.log(chalk.cyan('                    Ruvector Packages'));
      console.log(chalk.cyan('═══════════════════════════════════════════════════════════════\n'));

      const categories = {
        core: { title: '📦 Core Packages', packages: [] },
        tools: { title: '🔧 Tools & Extensions', packages: [] },
        platform: { title: '🖥️  Platform Bindings', packages: [] },
        legacy: { title: '📜 Legacy Packages', packages: [] }
      };

      // Group by category
      Object.entries(availablePackages).forEach(([key, pkg]) => {
        if (categories[pkg.category]) {
          categories[pkg.category].packages.push({ key, ...pkg });
        }
      });

      // Display by category
      for (const [catKey, cat] of Object.entries(categories)) {
        if (cat.packages.length === 0) continue;

        console.log(chalk.cyan(`${cat.title}`));
        console.log(chalk.gray('─'.repeat(60)));

        cat.packages.forEach(pkg => {
          const status = pkg.installed ? chalk.green('✓') : chalk.gray('○');
          const statusText = pkg.installed ? chalk.green('installed') : chalk.gray('available');
          console.log(chalk.white(`  ${status} ${chalk.yellow(pkg.key.padEnd(18))} ${statusText}`));
          console.log(chalk.gray(`      ${pkg.description}`));
          console.log(chalk.gray(`      npm: ${chalk.white(pkg.name)}\n`));
        });
      }

      console.log(chalk.cyan('═══════════════════════════════════════════════════════════════'));
      console.log(chalk.cyan('Usage:'));
      console.log(chalk.white('  npx ruvector install gnn              # Install GNN package'));
      console.log(chalk.white('  npx ruvector install graph-node       # Install graph database'));
      console.log(chalk.white('  npx ruvector install agentic-synth    # Install data generator'));
      console.log(chalk.white('  npx ruvector install --all            # Install all core packages'));
      console.log(chalk.white('  npx ruvector install -i               # Interactive selection'));
      console.log(chalk.gray('\n  Note: Platform bindings are auto-detected by @ruvector/core'));
      return;
    }

    // Interactive mode
    if (options.interactive) {
      const readline = require('readline');
      const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
      });

      console.log(chalk.cyan('\nSelect packages to install:\n'));

      const notInstalled = Object.entries(availablePackages)
        .filter(([_, pkg]) => !pkg.installed);

      if (notInstalled.length === 0) {
        console.log(chalk.green('All packages are already installed!'));
        rl.close();
        return;
      }

      notInstalled.forEach(([key, pkg], i) => {
        console.log(chalk.white(`  ${i + 1}. ${chalk.yellow(key)} - ${pkg.description}`));
      });
      console.log(chalk.white(`  ${notInstalled.length + 1}. ${chalk.yellow('all')} - Install all packages`));
      console.log(chalk.white(`  0. ${chalk.gray('cancel')} - Exit without installing`));

      rl.question(chalk.cyan('\nEnter selection (comma-separated for multiple): '), (answer) => {
        rl.close();

        const selections = answer.split(',').map(s => s.trim());
        let toInstall = [];

        for (const sel of selections) {
          if (sel === '0' || sel.toLowerCase() === 'cancel') {
            console.log(chalk.yellow('Installation cancelled.'));
            return;
          }
          if (sel === String(notInstalled.length + 1) || sel.toLowerCase() === 'all') {
            toInstall = notInstalled.map(([_, pkg]) => pkg.name);
            break;
          }
          const idx = parseInt(sel) - 1;
          if (idx >= 0 && idx < notInstalled.length) {
            toInstall.push(notInstalled[idx][1].name);
          }
        }

        if (toInstall.length === 0) {
          console.log(chalk.yellow('No valid packages selected.'));
          return;
        }

        installPackages(toInstall);
      });
      return;
    }

    // Install all (core + tools only, not platform-specific or legacy)
    if (options.all) {
      const toInstall = Object.values(availablePackages)
        .filter(pkg => !pkg.installed && (pkg.category === 'core' || pkg.category === 'tools'))
        .map(pkg => pkg.name);

      if (toInstall.length === 0) {
        console.log(chalk.green('All core packages are already installed!'));
        return;
      }

      console.log(chalk.cyan(`Installing ${toInstall.length} packages...`));
      installPackages(toInstall);
      return;
    }

    // Install specific packages
    const toInstall = [];
    for (const pkg of packages) {
      const key = pkg.toLowerCase().replace('@ruvector/', '');
      if (availablePackages[key]) {
        if (availablePackages[key].installed) {
          console.log(chalk.yellow(`${availablePackages[key].name} is already installed`));
        } else {
          toInstall.push(availablePackages[key].name);
        }
      } else {
        console.log(chalk.red(`Unknown package: ${pkg}`));
        console.log(chalk.gray(`Available: ${Object.keys(availablePackages).join(', ')}`));
      }
    }

    if (toInstall.length > 0) {
      installPackages(toInstall);
    }

    function installPackages(pkgs) {
      const spinner = ora(`Installing ${pkgs.join(', ')}...`).start();

      try {
        // Detect package manager
        let pm = 'npm';
        if (fs.existsSync('yarn.lock')) pm = 'yarn';
        else if (fs.existsSync('pnpm-lock.yaml')) pm = 'pnpm';
        else if (fs.existsSync('bun.lockb')) pm = 'bun';

        const cmd = pm === 'yarn' ? `yarn add ${pkgs.join(' ')}`
                  : pm === 'pnpm' ? `pnpm add ${pkgs.join(' ')}`
                  : pm === 'bun' ? `bun add ${pkgs.join(' ')}`
                  : `npm install ${pkgs.join(' ')}`;

        execSync(cmd, { stdio: 'pipe' });

        spinner.succeed(chalk.green(`Installed: ${pkgs.join(', ')}`));
        console.log(chalk.cyan('\nRun "npx ruvector info" to verify installation.'));
      } catch (error) {
        spinner.fail(chalk.red('Installation failed'));
        console.error(chalk.red(error.message));
        console.log(chalk.yellow(`\nTry manually: npm install ${pkgs.join(' ')}`));
        process.exit(1);
      }
    }
  });

// =============================================================================
// GNN Commands
// =============================================================================

// Helper to check GNN availability
function requireGnn() {
  if (!gnnAvailable) {
    console.error(chalk.red('Error: GNN module not available.'));
    console.error(chalk.yellow('Install it with: npm install @ruvector/gnn'));
    process.exit(1);
  }
}

// GNN parent command
const gnnCmd = program
  .command('gnn')
  .description('Graph Neural Network operations');

// GNN Layer command
gnnCmd
  .command('layer')
  .description('Create and test a GNN layer')
  .requiredOption('-i, --input-dim <number>', 'Input dimension')
  .requiredOption('-h, --hidden-dim <number>', 'Hidden dimension')
  .option('-a, --heads <number>', 'Number of attention heads', '4')
  .option('-d, --dropout <number>', 'Dropout rate', '0.1')
  .option('--test', 'Run a test forward pass')
  .option('-o, --output <file>', 'Save layer config to JSON file')
  .action((options) => {
    requireGnn();
    const spinner = ora('Creating GNN layer...').start();

    try {
      const inputDim = parseInt(options.inputDim);
      const hiddenDim = parseInt(options.hiddenDim);
      const heads = parseInt(options.heads);
      const dropout = parseFloat(options.dropout);

      const layer = new RuvectorLayer(inputDim, hiddenDim, heads, dropout);
      spinner.succeed(chalk.green('GNN Layer created'));

      console.log(chalk.cyan('\nLayer Configuration:'));
      console.log(chalk.white(`  Input Dim:  ${chalk.yellow(inputDim)}`));
      console.log(chalk.white(`  Hidden Dim: ${chalk.yellow(hiddenDim)}`));
      console.log(chalk.white(`  Heads:      ${chalk.yellow(heads)}`));
      console.log(chalk.white(`  Dropout:    ${chalk.yellow(dropout)}`));

      if (options.test) {
        spinner.start('Running test forward pass...');

        // Create test data
        const nodeEmbedding = Array.from({ length: inputDim }, () => Math.random());
        const neighborEmbeddings = [
          Array.from({ length: inputDim }, () => Math.random()),
          Array.from({ length: inputDim }, () => Math.random())
        ];
        const edgeWeights = [0.6, 0.4];

        const output = layer.forward(nodeEmbedding, neighborEmbeddings, edgeWeights);
        spinner.succeed(chalk.green('Forward pass completed'));

        console.log(chalk.cyan('\nTest Results:'));
        console.log(chalk.white(`  Input shape:  ${chalk.yellow(`[${inputDim}]`)}`));
        console.log(chalk.white(`  Output shape: ${chalk.yellow(`[${output.length}]`)}`));
        console.log(chalk.white(`  Output sample: ${chalk.gray(`[${output.slice(0, 4).map(v => v.toFixed(4)).join(', ')}...]`)}`));
      }

      if (options.output) {
        const config = layer.toJson();
        fs.writeFileSync(options.output, config);
        console.log(chalk.green(`\nLayer config saved to: ${options.output}`));
      }
    } catch (error) {
      spinner.fail(chalk.red('Failed to create GNN layer'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// GNN Compress command
gnnCmd
  .command('compress')
  .description('Compress embeddings using adaptive tensor compression')
  .requiredOption('-f, --file <path>', 'Input JSON file with embeddings')
  .option('-l, --level <type>', 'Compression level (none|half|pq8|pq4|binary)', 'auto')
  .option('-a, --access-freq <number>', 'Access frequency for auto compression (0.0-1.0)', '0.5')
  .option('-o, --output <file>', 'Output file for compressed data')
  .action((options) => {
    requireGnn();
    const spinner = ora('Loading embeddings...').start();

    try {
      const data = JSON.parse(fs.readFileSync(options.file, 'utf8'));
      const embeddings = Array.isArray(data) ? data : [data];

      spinner.text = 'Compressing embeddings...';
      const compressor = new TensorCompress();
      const accessFreq = parseFloat(options.accessFreq);

      const results = [];
      let totalOriginalSize = 0;
      let totalCompressedSize = 0;

      for (const embedding of embeddings) {
        const vec = embedding.vector || embedding;
        totalOriginalSize += vec.length * 4; // float32 = 4 bytes

        let compressed;
        if (options.level === 'auto') {
          compressed = compressor.compress(vec, accessFreq);
        } else {
          const levelConfig = { levelType: options.level };
          if (options.level === 'pq8') {
            levelConfig.subvectors = 8;
            levelConfig.centroids = 256;
          } else if (options.level === 'pq4') {
            levelConfig.subvectors = 8;
          }
          compressed = compressor.compressWithLevel(vec, levelConfig);
        }

        totalCompressedSize += compressed.length;
        results.push({
          id: embedding.id,
          compressed
        });
      }

      const ratio = (totalOriginalSize / totalCompressedSize).toFixed(2);
      const savings = ((1 - totalCompressedSize / totalOriginalSize) * 100).toFixed(1);

      spinner.succeed(chalk.green(`Compressed ${embeddings.length} embeddings`));

      console.log(chalk.cyan('\nCompression Results:'));
      console.log(chalk.white(`  Embeddings:    ${chalk.yellow(embeddings.length)}`));
      console.log(chalk.white(`  Level:         ${chalk.yellow(options.level === 'auto' ? `auto (${getCompressionLevel(accessFreq)})` : options.level)}`));
      console.log(chalk.white(`  Original:      ${chalk.yellow((totalOriginalSize / 1024).toFixed(2) + ' KB')}`));
      console.log(chalk.white(`  Compressed:    ${chalk.yellow((totalCompressedSize / 1024).toFixed(2) + ' KB')}`));
      console.log(chalk.white(`  Ratio:         ${chalk.yellow(ratio + 'x')}`));
      console.log(chalk.white(`  Savings:       ${chalk.yellow(savings + '%')}`));

      if (options.output) {
        fs.writeFileSync(options.output, JSON.stringify(results, null, 2));
        console.log(chalk.green(`\nCompressed data saved to: ${options.output}`));
      }
    } catch (error) {
      spinner.fail(chalk.red('Failed to compress embeddings'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// GNN Search command
gnnCmd
  .command('search')
  .description('Differentiable search with soft attention')
  .requiredOption('-q, --query <json>', 'Query vector as JSON array')
  .requiredOption('-c, --candidates <file>', 'Candidates file (JSON array of vectors)')
  .option('-k, --top-k <number>', 'Number of results', '5')
  .option('-t, --temperature <number>', 'Softmax temperature (lower=sharper)', '1.0')
  .action((options) => {
    requireGnn();
    const spinner = ora('Loading candidates...').start();

    try {
      const query = JSON.parse(options.query);
      const candidatesData = JSON.parse(fs.readFileSync(options.candidates, 'utf8'));
      const candidates = candidatesData.map(c => c.vector || c);
      const k = parseInt(options.topK);
      const temperature = parseFloat(options.temperature);

      spinner.text = 'Running differentiable search...';
      const result = differentiableSearch(query, candidates, k, temperature);

      spinner.succeed(chalk.green(`Found top-${k} results`));

      console.log(chalk.cyan('\nSearch Results:'));
      console.log(chalk.white(`  Query dim:     ${chalk.yellow(query.length)}`));
      console.log(chalk.white(`  Candidates:    ${chalk.yellow(candidates.length)}`));
      console.log(chalk.white(`  Temperature:   ${chalk.yellow(temperature)}`));

      console.log(chalk.cyan('\nTop-K Results:'));
      for (let i = 0; i < result.indices.length; i++) {
        const idx = result.indices[i];
        const weight = result.weights[i];
        const id = candidatesData[idx]?.id || `candidate_${idx}`;
        console.log(chalk.white(`  ${i + 1}. ${chalk.yellow(id)} (index: ${idx})`));
        console.log(chalk.gray(`     Weight: ${weight.toFixed(6)}`));
      }
    } catch (error) {
      spinner.fail(chalk.red('Failed to run search'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// GNN Info command
gnnCmd
  .command('info')
  .description('Show GNN module information')
  .action(() => {
    if (!gnnAvailable) {
      console.log(chalk.yellow('\nGNN Module: Not installed'));
      console.log(chalk.white('Install with: npm install @ruvector/gnn'));
      return;
    }

    console.log(chalk.cyan('\nGNN Module Information'));
    console.log(chalk.white(`  Status:         ${chalk.green('Available')}`));
    console.log(chalk.white(`  Platform:       ${chalk.yellow(process.platform)}`));
    console.log(chalk.white(`  Architecture:   ${chalk.yellow(process.arch)}`));

    console.log(chalk.cyan('\nAvailable Features:'));
    console.log(chalk.white(`  • RuvectorLayer   - GNN layer with multi-head attention`));
    console.log(chalk.white(`  • TensorCompress  - Adaptive tensor compression (5 levels)`));
    console.log(chalk.white(`  • differentiableSearch - Soft attention-based search`));
    console.log(chalk.white(`  • hierarchicalForward  - Multi-layer GNN processing`));

    console.log(chalk.cyan('\nCompression Levels:'));
    console.log(chalk.gray(`  none   (freq > 0.8)  - Full precision, hot data`));
    console.log(chalk.gray(`  half   (freq > 0.4)  - ~50% savings, warm data`));
    console.log(chalk.gray(`  pq8    (freq > 0.1)  - ~8x compression, cool data`));
    console.log(chalk.gray(`  pq4    (freq > 0.01) - ~16x compression, cold data`));
    console.log(chalk.gray(`  binary (freq <= 0.01) - ~32x compression, archive`));
  });

program.parse();
