import eslint from '@eslint/js';
import tseslintPlugin from '@typescript-eslint/eslint-plugin';
import tseslintParser from '@typescript-eslint/parser';

export default [
  eslint.configs.recommended,
  {
    files: ['**/*.ts', '**/*.tsx'],
    languageOptions: {
      parser: tseslintParser,
      parserOptions: {
        project: './tsconfig.json',
      },
    },
    plugins: {
      '@typescript-eslint': tseslintPlugin,
    },
    rules: {
      // Unused variables as warnings (not errors)
      '@typescript-eslint/no-unused-vars': ['warn', {
        argsIgnorePattern: '^_',
        varsIgnorePattern: '^_',
        caughtErrorsIgnorePattern: '^_',
      }],
      // Disable base rule to avoid conflicts
      'no-unused-vars': 'off',
      
      // Explicit any as warning (not error)
      '@typescript-eslint/no-explicit-any': 'warn',
      
      // Allow require imports (for compatibility)
      '@typescript-eslint/no-require-imports': 'off',
      
      // Disable no-undef for TypeScript (TypeScript handles this)
      'no-undef': 'off',
    },
  },
  {
    ignores: [
      'dist/**',
      'node_modules/**',
      '**/*.js',
      '**/*.cjs',
      '**/*.mjs',
      'public/**',
      'ml/**',
      'data/**',
      'scripts/**',
      'src/device/**',
      'src/web/**',
      'tests/**',
      'playwright.config.ts',
    ],
  }
];
