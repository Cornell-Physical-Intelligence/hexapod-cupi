import React from 'react';
import activeModel from '../../robot/active_model.json';

// The same detailed CAD inspector is used by the repository quickstart.
// Historical checkpoints keep their own explicit asset paths and task IDs.
export default function App() {
  // Vite's public directory serves the explicit file; a directory URL falls
  // through to the application shell and would recursively embed this page.
  const preview = '/' + activeModel.preview.replace(/^robot\//, '') + 'index.html';
  return <iframe title="Hexapod main robot: parts and joint travel"
    src={preview} style={{width:'100%',height:'100%',border:0,display:'block'}} />;
}
