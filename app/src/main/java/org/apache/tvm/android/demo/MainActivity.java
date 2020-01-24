/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.tvm.android.demo;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.content.DialogInterface;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Vector;

import org.apache.tvm.Function;
import org.apache.tvm.Module;
import org.apache.tvm.NDArray;
import org.apache.tvm.TVMContext;
import org.apache.tvm.TVMValue;
import org.apache.tvm.TVMType;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = MainActivity.class.getSimpleName();

    private static final int PERMISSIONS_REQUEST    = 100;
    private static final int PICTURE_FROM_GALLERY   = 101;
    private static final int PICTURE_FROM_CAMERA    = 102;
    private static int IMAGE_PREVIEW                = 720;
    private double IMAGE_PREVIEW_RATIO              = 1.0;

    // TVM constants
    private static final int OUTPUT_INDEX           = 0;
    private static final int IMG_CHANNEL            = 3;
    private static final String INPUT_NAME          = "input.1";

    // Configuration values for extraction model. Note that the graph, lib and params is not
    // included with TVM and must be manually placed in the assets/ directory by the user.
    // Graphs and models downloaded from https://github.com/pjreddie/darknet/blob/ may be
    // converted e.g. via  define_and_compile_model.py.
    private static final boolean EXE_GPU            = false;
    private static final int MODEL_INPUT_SIZE       = 256;
    private static final int MODEL_OUTPUT_SIZE      = 259;

    private static final String[] MODEL_NAME        = {"marvel", "matrix_world", "ocean", "picasso", "scream", "spiderman"};

    private Uri mCameraImageUri;
    private String mCameraImagePath;
    private ImageView mImageView;
    private TextView mResultView;
    private TextView mModelView;
    private AssetManager assetManager;
    private Module graphRuntimeModule;
    private Vector<String> labels = new Vector<String>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        assetManager = getAssets();

        mImageView = (ImageView) findViewById(R.id.imageView);
        mResultView = (TextView) findViewById(R.id.resultTextView);
        mModelView = (TextView) findViewById(R.id.modelTextView);
        findViewById(R.id.btnPickImage).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                showPictureDialog();
            }
        });
        findViewById(R.id.btnPickModel).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                showModelDialog();
            }
        });

        if (hasPermission()) {
            // instantiate tvm runtime and setup environment on background after application begin
            new LoadModelAsyncTask().execute(new Integer(0));
        } else {
            requestPermission();
        }
    }

    /*
        Load precompiled model on TVM graph runtime and init the system.
     */
    private class LoadModelAsyncTask extends AsyncTask<Integer, Void, Integer> {
        ProgressDialog dialog = new ProgressDialog(MainActivity.this);

        @Override
        protected Integer doInBackground(Integer... whichs) {
                int which = whichs[0].intValue();

                // load json graph
                String modelGraph = null;
                String graphFilename = MODEL_NAME[which] + ".json";
                Log.i(TAG, "Reading json graph from: " + graphFilename);
                try {
                    modelGraph = new String(getBytesFromFile(assetManager, graphFilename));
                } catch (IOException e) {
                    Log.e(TAG, "Problem reading json graph file!" + e);
                    return -1;//failure
                }
    
                // upload tvm compiled function on application cache folder
                String libCacheFilePath = null;
                String libFilename = EXE_GPU ? MODEL_NAME[which] + ".so" : MODEL_NAME[which] + ".so";
                Log.i(TAG, "Uploading compiled function to cache folder");
                try {
                    libCacheFilePath = getTempLibFilePath(libFilename);
                    byte[] modelLibByte = getBytesFromFile(assetManager, libFilename);
                    FileOutputStream fos = new FileOutputStream(libCacheFilePath);
                    fos.write(modelLibByte);
                    fos.close();
                } catch (IOException e) {
                    Log.e(TAG, "Problem uploading compiled function!" + e);
                    return -1;//failure
                }
    
                // load parameters
                byte[] modelParams = null;
                String paramFilename = MODEL_NAME[which] + ".params";
                try {
                    modelParams = getBytesFromFile(assetManager, paramFilename);
                } catch (IOException e) {
                    Log.e(TAG, "Problem reading params file!" + e);
                    return -1;//failure
                }
    
                // create java tvm context
                TVMContext tvmCtx = EXE_GPU ? TVMContext.opencl() : TVMContext.cpu();
    
                // tvm module for compiled functions
                Module modelLib = Module.load(libCacheFilePath);
    
                // get global function module for graph runtime
                Function runtimeCreFun = Function.getFunction("tvm.graph_runtime.create");
                TVMValue runtimeCreFunRes = runtimeCreFun.pushArg(modelGraph)
                        .pushArg(modelLib)
                        .pushArg(tvmCtx.deviceType)
                        .pushArg(tvmCtx.deviceId)
                        .invoke();
                graphRuntimeModule = runtimeCreFunRes.asModule();
    
                // get the function from the module(load parameters)
                Function loadParamFunc = graphRuntimeModule.getFunction("load_params");
                loadParamFunc.pushArg(modelParams).invoke();
    
                // release tvm local variables
                modelLib.release();
                loadParamFunc.release();
                runtimeCreFun.release();
                String label = "Model: " + MODEL_NAME[which];
                mModelView.setText(label);
    
            return 0;//success
        }

        @Override
        protected void onPreExecute() {
            dialog.setCancelable(false);
            dialog.setMessage("Loading Model...");
            dialog.show();
            super.onPreExecute();
        }

        @Override
        protected void onPostExecute(Integer status) {
            if (dialog != null && dialog.isShowing()) {
                dialog.dismiss();
            }
            if (status != 0) {
                showDialog("Error", "Fail to initialized model, check compiled model");
            }
        }
    }

    public byte floatToByte (float num) {
           if (num > 255) return (byte) 255;
           if (num < 0) return (byte) 0;
           return (byte) num;
    }

    public synchronized Bitmap floatToBitmap(float[] img) {
        if (img != null) {
            Bitmap image = Bitmap.createBitmap(MODEL_OUTPUT_SIZE, MODEL_OUTPUT_SIZE, Bitmap.Config.ARGB_8888);
            int column = 0;
            int row = 0;
            while(row < MODEL_OUTPUT_SIZE)
            {
                 while(column < MODEL_OUTPUT_SIZE)
                 {
                     byte r = floatToByte(img[0*MODEL_OUTPUT_SIZE*MODEL_OUTPUT_SIZE+row*MODEL_OUTPUT_SIZE+column]);
                     byte g = floatToByte(img[1*MODEL_OUTPUT_SIZE*MODEL_OUTPUT_SIZE+row*MODEL_OUTPUT_SIZE+column]);
                     byte b = floatToByte(img[2*MODEL_OUTPUT_SIZE*MODEL_OUTPUT_SIZE+row*MODEL_OUTPUT_SIZE+column]);
                     image.setPixel(column, row, Color.rgb(r, g, b));
                     column++;
                 }
                 column = 0;
                 row++;
            }
            return image;
        }
        return null;
     }

    /*
        Execute prediction for processed decode input bitmap image content on TVM graph runtime.
     */
    private class ModelRunAsyncTask extends AsyncTask<Bitmap, Void, Bitmap> {
        ProgressDialog dialog = new ProgressDialog(MainActivity.this);

        @Override
        protected Bitmap doInBackground(Bitmap... bitmaps) {
            if (null != graphRuntimeModule) {
                int count  = bitmaps.length;
                for (int i = 0 ; i < count ; i++) {
                    Log.i(TAG, "Decode JPEG image content");

                    // extract the jpeg content
                    ByteArrayOutputStream stream = new ByteArrayOutputStream();
                    bitmaps[i].compress(Bitmap.CompressFormat.JPEG,100,stream);
                    byte[] byteArray = stream.toByteArray();
                    Bitmap imageBitmap = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);

                    // crop input image at centre to model input size
                    // commecial deploy note:: instead of cropying image do resize
                    // image to model input size so we never lost the image content
                    Bitmap cropImageBitmap = Bitmap.createBitmap(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, Bitmap.Config.ARGB_8888);
                    Matrix frameToCropTransform = getTransformationMatrix(imageBitmap.getWidth(), imageBitmap.getHeight(),
                            MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 0, true);
                    Canvas canvas = new Canvas(cropImageBitmap);
                    canvas.drawBitmap(imageBitmap, frameToCropTransform, null);
                    // image pixel int values
                    int[] pixelValues = new int[MODEL_INPUT_SIZE * MODEL_INPUT_SIZE];
                    // image RGB float values
                    float[] imgRgbValues = new float[MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * IMG_CHANNEL];
                    // image RGB transpose float values
                    float[] imgRgbTranValues = new float[MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * IMG_CHANNEL];

                    // pre-process the image data from 0-255 int to normalized float based on the
                    // provided parameters.
                    cropImageBitmap.getPixels(pixelValues, 0, MODEL_INPUT_SIZE, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
                    for (int j = 0; j < pixelValues.length; ++j) {
                        imgRgbValues[j * 3 + 0] = ((pixelValues[j] >> 16) & 0xFF)/255.0f;
                        imgRgbValues[j * 3 + 1] = ((pixelValues[j] >> 8) & 0xFF)/255.0f;
                        imgRgbValues[j * 3 + 2] = (pixelValues[j] & 0xFF)/255.0f;
                    }

                    // pre-process the image rgb data transpose based on the provided parameters.
                    for (int k = 0; k < IMG_CHANNEL; ++k) {
                        for (int l = 0; l < MODEL_INPUT_SIZE; ++l) {
                            for (int m = 0; m < MODEL_INPUT_SIZE; ++m) {
                                int dst_index = m + MODEL_INPUT_SIZE*l + MODEL_INPUT_SIZE*MODEL_INPUT_SIZE*k;
                                int src_index = k + IMG_CHANNEL*m + IMG_CHANNEL*MODEL_INPUT_SIZE*l;
                                imgRgbTranValues[dst_index] = imgRgbValues[src_index];
                            }
                        }
                    }

                    // get the function from the module(set input data)
                    Log.i(TAG, "set input data");
                    NDArray inputNdArray = NDArray.empty(new long[]{1, IMG_CHANNEL, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE}, new TVMType("float32"));
                    inputNdArray.copyFrom(imgRgbTranValues);
                    Function setInputFunc = graphRuntimeModule.getFunction("set_input");
                    setInputFunc.pushArg(INPUT_NAME).pushArg(inputNdArray).invoke();
                    // release tvm local variables
                    inputNdArray.release();
                    setInputFunc.release();

                    // get the function from the module(run it)
                    Log.i(TAG, "run function on target");
                    Function runFunc = graphRuntimeModule.getFunction("run");
                    long processingTimeMs = SystemClock.uptimeMillis();
                    runFunc.invoke();
                    processingTimeMs = SystemClock.uptimeMillis() - processingTimeMs;
                    // release tvm local variables
                    runFunc.release();

                    // get the function from the module(get output data)
                    Log.i(TAG, "get output data");
                    NDArray outputNdArray = NDArray.empty(new long[]{1, IMG_CHANNEL, MODEL_OUTPUT_SIZE, MODEL_OUTPUT_SIZE}, new TVMType("float32"));
                    Function getOutputFunc = graphRuntimeModule.getFunction("get_output");
                    getOutputFunc.pushArg(OUTPUT_INDEX).pushArg(outputNdArray).invoke();
                    Bitmap output = floatToBitmap(outputNdArray.asFloatArray());
                    // release tvm local variables
                    outputNdArray.release();
                    getOutputFunc.release();
                    int IMAGE_PREVIEW_HEIGHT = (int) ( (double) IMAGE_PREVIEW * IMAGE_PREVIEW_RATIO);
                    Bitmap scaled = Bitmap.createScaledBitmap(output, IMAGE_PREVIEW, IMAGE_PREVIEW_HEIGHT, true);

                    // display the result from extracted output data
                    if (null != output) {
                        String label = "Running Time : " + processingTimeMs + "ms";
                        mResultView.setText(label);
                    }
                    Log.i(TAG, "prediction finished");
                    return scaled;
                }
                return null;
            }
            return null;
        }

        @Override
        protected void onPreExecute() {
            dialog.setCancelable(false);
            dialog.setMessage("Model running on image...");
            dialog.show();
            super.onPreExecute();
        }

        @Override
        protected void onPostExecute(Bitmap image) {
            if (dialog != null && dialog.isShowing()) {
                mImageView.setImageBitmap(image);
                dialog.dismiss();
            }
            if (image == null) {
                showDialog("Error", "Fail to predict image, GraphRuntime exception");
            }
        }
    }

    @Override
    protected void onDestroy() {
        // release tvm local variables
        if (null != graphRuntimeModule)
            graphRuntimeModule.release();
        super.onDestroy();
    }

    /**
     * Read file from assets and return byte array.
     *
     * @param assets The asset manager to be used to load assets.
     * @param fileName The filepath of read file.
     * @return byte[] file content
     * @throws IOException
     */
    private byte[] getBytesFromFile(AssetManager assets, String fileName) throws IOException {
        InputStream is = assets.open(fileName);
        int length = is.available();
        byte[] bytes = new byte[length];
        // Read in the bytes
        int offset = 0;
        int numRead = 0;
        try {
            while (offset < bytes.length
                    && (numRead = is.read(bytes, offset, bytes.length - offset)) >= 0) {
                offset += numRead;
            }
        } finally {
            is.close();
        }
        // Ensure all the bytes have been read in
        if (offset < bytes.length) {
            throw new IOException("Could not completely read file " + fileName);
        }
        return bytes;
    }

    private void showModelDialog(){
        AlertDialog.Builder modelDialog = new AlertDialog.Builder(this);
        modelDialog.setTitle("Select Model");
        modelDialog.setItems(MODEL_NAME,
                new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                         new LoadModelAsyncTask().execute(new Integer(which));
                    }
                });
        modelDialog.show();
    }

    /**
     * Dialog show pick option for select image from Gallery or Camera.
     */
    private void showPictureDialog(){
        AlertDialog.Builder pictureDialog = new AlertDialog.Builder(this);
        pictureDialog.setTitle("Select Action");
        String[] pictureDialogItems = {
                "Select photo from gallery",
                "Capture photo from camera" };
        pictureDialog.setItems(pictureDialogItems,
                new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        switch (which) {
                            case 0:
                                choosePhotoFromGallery();
                                break;
                            case 1:
                                takePhotoFromCamera();
                                break;
                        }
                    }
                });
        pictureDialog.show();
    }

    /**
     * Request to pick image from Gallery.
     */
    public void choosePhotoFromGallery() {
        Intent galleryIntent = new Intent(Intent.ACTION_PICK,
                android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);

        startActivityForResult(galleryIntent, PICTURE_FROM_GALLERY);
    }

    /**
     * Request to capture image from Camera.
     */
    private void takePhotoFromCamera() {
        Intent intent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);

        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.N) {
            File file = createImageFile();
            mCameraImagePath = file.getPath();
            mCameraImageUri = Uri.fromFile(file);
            Log.i(TAG, "[kk] <");
        } else {
            File file = new File(createImageFile().getPath());
            mCameraImageUri = FileProvider.getUriForFile(getApplicationContext(), getApplicationContext().getPackageName() + ".provider", file);
            mCameraImagePath = file.getPath();
            Log.i(TAG, "[kk] >");
        }
        intent.putExtra(MediaStore.EXTRA_OUTPUT, mCameraImageUri);
        startActivityForResult(intent, PICTURE_FROM_CAMERA);
    }

    public static Bitmap rotateImage(Bitmap source, float angle) {
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(),
                                   matrix, true);
    }
    public String UriToPath(Uri uri) {
        String path = "";
        if (getContentResolver() != null) {
            String [] proj={MediaStore.Images.Media.DATA};
            Cursor cursor = getContentResolver().query(uri, proj,  null, null, null);
            if (cursor != null) {
                cursor.moveToFirst();
                int idx = cursor.getColumnIndex(MediaStore.Images.ImageColumns.DATA);
                path = cursor.getString(idx);
                cursor.close();
            }
        }
        return path;
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == this.RESULT_CANCELED) {
            return;
        }
        Uri contentURI = null;
        String contentPath = "";
        if (requestCode == PICTURE_FROM_GALLERY) {
            if (data != null) {
                contentURI = data.getData();
                contentPath = UriToPath(contentURI);
            }
        } else if (requestCode == PICTURE_FROM_CAMERA) {
            contentURI = mCameraImageUri;
            contentPath = mCameraImagePath;
        }
        if (null != contentURI) {
            try {
                Bitmap bitmap_ori = MediaStore.Images.Media.getBitmap(this.getContentResolver(), contentURI);
                Log.i(TAG, "[IMG] PATH" + contentPath);
                ExifInterface ei = new ExifInterface(contentPath);
                int orientation = ei.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED);
                Bitmap bitmap = null;
                switch(orientation) {
                    case ExifInterface.ORIENTATION_ROTATE_90:
                        bitmap = rotateImage(bitmap_ori, 90);
                        break;
                    case ExifInterface.ORIENTATION_ROTATE_180:
                        bitmap = rotateImage(bitmap_ori, 180);
                        break;
                    case ExifInterface.ORIENTATION_ROTATE_270:
                        bitmap = rotateImage(bitmap_ori, 270);
                        break;
                    case ExifInterface.ORIENTATION_NORMAL:
                    default:
                        bitmap = bitmap_ori;
                }
                IMAGE_PREVIEW_RATIO = (double) bitmap.getHeight() / bitmap.getWidth();
                Bitmap scaled = Bitmap.createScaledBitmap(bitmap, IMAGE_PREVIEW, IMAGE_PREVIEW, true);
                mImageView.setImageBitmap(scaled);
                new ModelRunAsyncTask().execute(scaled);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * Get application cache path where to place compiled functions.
     *
     * @param fileName library file name.
     * @return String application cache folder path
     * @throws IOException
     */
    private final String getTempLibFilePath(String fileName) throws IOException {
        File tempDir = File.createTempFile("tvm4j_demo_", "");
        if (!tempDir.delete() || !tempDir.mkdir()) {
            throw new IOException("Couldn't create directory " + tempDir.getAbsolutePath());
        }
        return (tempDir + File.separator + fileName);
    }

    /**
     * Create image file under storage where camera application save captured image.
     *
     * @return File image file under sdcard where camera can save image
     */
    private File createImageFile() {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_PICTURES);
        try {
            File image = File.createTempFile(
                    imageFileName,  // prefix
                    ".jpg",         // suffix
                    storageDir      // directory
            );
            return image;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    /**
     * Show dialog to user.
     *
     * @param title dialog display title
     * @param msg dialog display message
     */
    private void showDialog(String title, String msg) {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle(title);
        builder.setMessage(msg);
        builder.setCancelable(true);
        builder.setNeutralButton(android.R.string.ok,
                new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int id) {
                        dialog.cancel();
                        finish();
                    }
                });
        builder.create().show();
    }

    @Override
    public void onRequestPermissionsResult (final int requestCode, final String[] permissions, final int[] grantResults){
        if (requestCode == PERMISSIONS_REQUEST) {
            if (grantResults.length > 0
                    && grantResults[0] == PackageManager.PERMISSION_GRANTED
                    && grantResults[1] == PackageManager.PERMISSION_GRANTED) {
                // instantiate tvm runtime and setup environment on background after application begin
                new LoadModelAsyncTask().execute(new Integer(0));
            } else {
                requestPermission();
            }
        }
    }

    /**
     * Whether application has required mandatory permissions to run.
     */
    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED &&
                    checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    /**
     * Request required mandatory permission for application to run.
     */
    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA) ||
                    shouldShowRequestPermissionRationale(Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
                Toast.makeText(this,
                        "Camera AND storage permission are required for this demo", Toast.LENGTH_LONG).show();
            }
            requestPermissions(new String[] {Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE}, PERMISSIONS_REQUEST);
        }
    }

    /**
     * Returns a transformation matrix from one reference frame into another.
     * Handles cropping (if maintaining aspect ratio is desired) and rotation.
     *
     * @param srcWidth Width of source frame.
     * @param srcHeight Height of source frame.
     * @param dstWidth Width of destination frame.
     * @param dstHeight Height of destination frame.
     * @param applyRotation Amount of rotation to apply from one frame to another.
     *  Must be a multiple of 90.
     * @param maintainAspectRatio If true, will ensure that scaling in x and y remains constant,
     * cropping the image if necessary.
     * @return The transformation fulfilling the desired requirements.
     */
    public static Matrix getTransformationMatrix(
            final int srcWidth,
            final int srcHeight,
            final int dstWidth,
            final int dstHeight,
            final int applyRotation,
            final boolean maintainAspectRatio) {
        final Matrix matrix = new Matrix();

        if (applyRotation != 0) {
            if (applyRotation % 90 != 0) {
                Log.w(TAG, "Rotation of %d % 90 != 0 " + applyRotation);
            }

            // Translate so center of image is at origin.
            matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f);

            // Rotate around origin.
            matrix.postRotate(applyRotation);
        }

        // Account for the already applied rotation, if any, and then determine how
        // much scaling is needed for each axis.
        final boolean transpose = (Math.abs(applyRotation) + 90) % 180 == 0;

        final int inWidth = transpose ? srcHeight : srcWidth;
        final int inHeight = transpose ? srcWidth : srcHeight;

        // Apply scaling if necessary.
        if (inWidth != dstWidth || inHeight != dstHeight) {
            final float scaleFactorX = dstWidth / (float) inWidth;
            final float scaleFactorY = dstHeight / (float) inHeight;

            if (maintainAspectRatio) {
                // Scale by minimum factor so that dst is filled completely while
                // maintaining the aspect ratio. Some image may fall off the edge.
                final float scaleFactor = Math.max(scaleFactorX, scaleFactorY);
                matrix.postScale(scaleFactor, scaleFactor);
            } else {
                // Scale exactly to fill dst from src.
                matrix.postScale(scaleFactorX, scaleFactorY);
            }
        }

        if (applyRotation != 0) {
            // Translate back from origin centered reference to destination frame.
            matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f);
        }

        return matrix;
    }
}
