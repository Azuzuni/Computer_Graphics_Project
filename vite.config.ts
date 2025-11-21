import { defineConfig } from 'vite';
import * as path from 'path';

export default defineConfig({
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
      '@object': path.resolve(__dirname,'src/stage/Object.d.ts'),
    }
  }
});