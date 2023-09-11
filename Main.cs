using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace Feature_Matching
{
    public partial class Main : Form
    {
        private readonly ROI roi = new ROI();
        private readonly VideoCapture capture;
        private bool isCamOpen = false;

        public Main()
        {
            InitializeComponent();
            InitializeControl();
            WindowState = FormWindowState.Maximized;
            capture = new VideoCapture(0);
            capture.SetCaptureProperty(CapProp.FrameWidth, 1280);
            capture.SetCaptureProperty(CapProp.FrameHeight, 720);
        }

        private void InitializeControl()
        {
            toolStrip1.Renderer = new RemoveBorder();
            toolStrip2.Renderer = new RemoveBorder();
            toolStrip3.Renderer = new RemoveBorder();
            toolStrip4.Renderer = new RemoveBorder();
            toolStrip5.Renderer = new RemoveBorder();
            toolStrip6.Renderer = new RemoveBorder();
            toolStrip7.Renderer = new RemoveBorder();
        }

        private void StartCamera(object sender, EventArgs args)
        {
            isCamOpen = true;
            label2.Text = string.Empty;
            label3.Text = string.Empty;
            label4.Text = string.Empty;
            pictureBox1.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBox1.Dock = DockStyle.Fill;
            timer1.Start();
        }

        private void ProcessFrame(object sender, EventArgs args)
        {
            if (!isCamOpen) { return; }
            pictureBox1.Image?.Dispose();
            pictureBox1.Image = capture.QueryFrame().ToBitmap();
        }

        private void FeatureMatchingClick(object sender, EventArgs e)
        {
            if(isCamOpen)
            {
                isCamOpen= false;
                pictureBox1.SizeMode = PictureBoxSizeMode.AutoSize;
                pictureBox1.Dock = DockStyle.None;
                Image<Bgr, byte> inputImg = new Bitmap(pictureBox1.Image).ToImage<Bgr, byte>();
                AddImage(inputImg, "Input");
            }
            if(pictureBox1.Image == null || !roi.imgList.ContainsKey("Input") || !roi.imgList.ContainsKey("ROI 1"))
            {
                return;
            }
            Image<Bgr, byte> imgScene = roi.imgList["Input"].Clone();
            Image<Gray, byte> imgTemplate = roi.imgList["ROI 1"].Convert<Gray, byte>();
            Point[] points = ProcessImage(imgTemplate, imgScene.Convert<Gray, byte>());
            if(points != null)
            {
                Point centroid = roi.FindCentroid(points);
                roi.subsequentCentroid.Add(centroid);
                CvInvoke.Polylines(imgScene, points, true, new MCvScalar(255, 255, 0));
                CvInvoke.DrawMarker(imgScene, centroid, new MCvScalar(255, 255, 0), MarkerTypes.TiltedCross, 10);
            }
            if (roi.imgList.ContainsKey("ROI 2"))
            {
                Image<Gray, byte> imgTemplate2 = roi.imgList["ROI 2"].Convert<Gray, byte>();
                points = ProcessImage(imgTemplate2, imgScene.Convert<Gray, byte>());
                if (points != null)
                {
                    Point centroid = roi.FindCentroid(points);
                    roi.subsequentCentroid2.Add(centroid);
                    CvInvoke.Polylines(imgScene, points, true, new MCvScalar(255, 255, 0));
                    CvInvoke.DrawMarker(imgScene, centroid, new MCvScalar(255, 255, 0), MarkerTypes.TiltedCross, 10);
                }
                string error = roi.CheckTranslationRotation().Replace("\r\n", "*");
                string[] temp = error.Split('*');
                if (temp.Length > 0)
                {
                    label2.Text = temp[0];
                    if (temp.Length > 1)
                    {
                        label3.Text = temp[1];
                    }
                }
                else
                {
                    label2.Text = "";
                    label3.Text = "";
                }
            }
            pictureBox1.Image = imgScene.ToBitmap();
            GC.Collect();
        }

        private Point[] ProcessImage(Image<Gray, byte> imgTemplate, Image<Gray, byte> imgScene)
        {
            try
            {
                Mat img = new Mat();
                Mat homography = null;
                Point[] points = null;
                VectorOfKeyPoint templateKeyPoints = new VectorOfKeyPoint();
                VectorOfKeyPoint sceneKeyPoints = new VectorOfKeyPoint();
                VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();
                Mat templateDescriptor = new Mat();
                Mat sceneDescriptor = new Mat();
                Mat mask = new Mat();
                int k = 2;
                double uniquenessThreshold = 0.80;

                ORBDetector featureDetector = new ORBDetector(9000, 1.5f, 4);
                featureDetector.DetectAndCompute(imgTemplate, null, templateKeyPoints, templateDescriptor, false);
                featureDetector.DetectAndCompute(imgScene, null, sceneKeyPoints, sceneDescriptor, false);
                BFMatcher matcher = new BFMatcher(DistanceType.Hamming, false);
                matcher.Add(templateDescriptor);
                matcher.KnnMatch(sceneDescriptor, matches, k);

                mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
                mask.SetTo(new MCvScalar(255));
                Features2DToolbox.VoteForUniqueness(matches, uniquenessThreshold, mask);

                // Show Process
                Features2DToolbox.DrawMatches(imgTemplate, templateKeyPoints, imgScene, sceneKeyPoints, matches, img, new MCvScalar(255, 0, 0), new MCvScalar(0, 0, 255), mask);
                pictureBox1.Image = img.ToBitmap();
                MessageBox.Show("Feature Matches");

                int nonZeroCount = CvInvoke.CountNonZero(mask);
                if (nonZeroCount >= 4)
                {
                    nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(templateKeyPoints, sceneKeyPoints, matches, mask, 1.5, 20);
                    if (nonZeroCount >= 4)
                    {
                        homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(templateKeyPoints, sceneKeyPoints, matches, mask, 2);
                    }
                }
                if (homography != null)
                {
                    Rectangle rect = new Rectangle(Point.Empty, imgTemplate.Size);
                    PointF[] pts = new PointF[]
                    {
                    new PointF(rect.Left, rect.Bottom),
                    new PointF(rect.Right, rect.Bottom),
                    new PointF(rect.Right, rect.Top),
                    new PointF(rect.Left, rect.Top),
                    };
                    pts = CvInvoke.PerspectiveTransform(pts, homography);
                    points = Array.ConvertAll(pts, Point.Round);
                }
                label4.ForeColor = Color.LightGray;
                label4.Text = " Features Matched\n Successfully";
                return points;
            }
            catch
            {
                label4.ForeColor = Color.Red;
                label4.Text = " No Feature Found";
                return null;
            }
        }

        private void LoadImage(object sender, EventArgs e)
        {
            capture.Stop();
            OpenFileDialog file = new OpenFileDialog()
            {
                Title = "Load Image",
                Filter = "PNG Image|*.png|Bitmap Image|*.bmp|JPEG Image|*.jpeg|JPG Image|*.jpg|All files|*.*",
                FilterIndex = 4,
                RestoreDirectory = true
            };
            if (file.ShowDialog() == DialogResult.OK)
            {
                isCamOpen = false;
                timer1.Stop();
                label2.Text = string.Empty;
                label3.Text = string.Empty;
                label4.Text = string.Empty;
                pictureBox1.SizeMode = PictureBoxSizeMode.AutoSize;
                pictureBox1.Dock = DockStyle.None;
                string filepath = file.FileName;
                if (File.Exists(filepath))
                {
                    Image<Bgr, byte> inputImg = new Image<Bgr, byte>(filepath);
                    AddImage(inputImg, "Input");
                    treeView1.SelectedNode = null;
                    pictureBox1.Size = inputImg.Size;
                    pictureBox1.Image = inputImg.ToBitmap();
                }
            }
        }

        private void CaptureImage(object sender, EventArgs e)
        {
            if (isCamOpen == false) return;
            isCamOpen = false;
            pictureBox1.SizeMode = PictureBoxSizeMode.AutoSize;
            pictureBox1.Dock = DockStyle.None;
            Image<Bgr, byte> inputImg = new Bitmap(pictureBox1.Image).ToImage<Bgr, byte>();
            AddImage(inputImg, "Input");
            treeView1.SelectedNode = null;
            pictureBox1.Image = inputImg.ToBitmap();
        }

        private void AddImage(Image<Bgr, byte> img, string keyName) 
        { 
            if(!treeView1.Nodes.ContainsKey(keyName))
            {
                TreeNode node = new TreeNode(keyName)
                {
                    Name = keyName
                };
                treeView1.Nodes.Add(node);
                treeView1.SelectedNode = node;
            }
            if(!roi.imgList.ContainsKey(keyName))
            {
                roi.imgList.Add(keyName, img);
            }
            else
            {
                roi.imgList[keyName] = img;
            }
        }

        private void PictureBox1MouseDown(object sender, MouseEventArgs e)
        {
            if(treeView1.Nodes.ContainsKey("Input") && !isCamOpen && e.Button == MouseButtons.Left)
            {
                roi.isSelect = true;
                roi.startCoord = e.Location;
                roi.endCoord = Point.Empty;
            }
            else
            {
                Refresh();
            }
        }

        private void PictureBox1MouseMove(object sender, MouseEventArgs e)
        {
            if(roi.isSelect && !isCamOpen)
            {
                int width = Math.Max(roi.startCoord.X, e.X) - Math.Min(roi.startCoord.X, e.X);
                int height = Math.Max(roi.startCoord.Y, e.Y) - Math.Min(roi.startCoord.Y, e.Y);
                roi.rect = new Rectangle(Math.Min(roi.startCoord.X, e.X), Math.Min(roi.startCoord.Y, e.Y), width, height);
                Refresh();
            }
        }

        private void PictureBox1Paint(object sender, PaintEventArgs e)
        {
            SolidBrush sb = new SolidBrush(Color.FromArgb(100, 0, 255, 255));
            if (roi.rect != null && roi.endCoord == Point.Empty)
            {
                Rectangle rectBorder = new Rectangle(roi.rect.X - 1, roi.rect.Y - 1, roi.rect.Width, roi.rect.Height);
                e.Graphics.DrawRectangle(Pens.Cyan, Rectangle.Round(rectBorder));
                e.Graphics.FillRectangle(sb, roi.rect);
            }
        }

        private void PictureBox1MouseUp(object sender, MouseEventArgs e)
        {
            if(roi.isSelect && !isCamOpen && e.Button == MouseButtons.Left)
            {
                roi.isSelect = false;
                roi.endCoord = e.Location;
            }
        }

        private void GetROI(object sender, EventArgs e)
        {
            if (pictureBox1.Image == null || roi.rect == Rectangle.Empty)
            {
                return;
            }
            roi.subsequentCentroid.Clear();
            Image<Bgr, byte> img = roi.imgList["Input"].Clone();
            img.ROI = roi.rect;
            Image<Bgr, byte> imgROI = img.Copy();
            img.ROI = Rectangle.Empty;
            Point[] points = ProcessImage(imgROI.Convert<Gray, byte>(), img.Convert<Gray, byte>());
            if (points != null)
            {
                Point centroid = roi.FindCentroid(points);
                roi.subsequentCentroid.Add(centroid);
                CvInvoke.Polylines(img, points, true, new MCvScalar(255, 255, 0));
                CvInvoke.DrawMarker(img, centroid, new MCvScalar(255, 255, 0), MarkerTypes.TiltedCross, 10);
                pictureBox1.Image = img.ToBitmap();
                if (roi.subsequentCentroid.Any() && roi.subsequentCentroid2.Any())
                {
                    roi.defaultSize = roi.FindXYOffset(roi.subsequentCentroid[0], roi.subsequentCentroid2[0]);
                }
            }
            AddImage(imgROI, "ROI 1");
            treeView1.SelectedNode = null;
            GC.Collect();
        }

        private void TreeViewNodeMouseClick(object sender, TreeNodeMouseClickEventArgs e)
        {
            pictureBox1.Image = roi.imgList[e.Node.Text].ToBitmap();
        }

        private void GetROI2(object sender, EventArgs e)
        {
            if (pictureBox1.Image == null || roi.rect == Rectangle.Empty)
            {
                return;
            }
            roi.subsequentCentroid2.Clear();
            Image<Bgr, byte> img = roi.imgList["Input"].Clone();
            img.ROI = roi.rect;
            Image<Bgr, byte> imgROI = img.Copy();
            img.ROI = Rectangle.Empty;
            Point[] points = ProcessImage(imgROI.Convert<Gray, byte>(), img.Convert<Gray, byte>());
            if (points != null)
            {
                Point centroid = roi.FindCentroid(points);
                roi.subsequentCentroid2.Add(centroid);
                CvInvoke.Polylines(img, points, true, new MCvScalar(255, 255, 0));
                CvInvoke.DrawMarker(img, centroid, new MCvScalar(255, 255, 0), MarkerTypes.TiltedCross, 10);
                if (roi.subsequentCentroid.Any() && roi.subsequentCentroid2.Any())
                {
                    roi.defaultSize = roi.FindXYOffset(roi.subsequentCentroid[0], roi.subsequentCentroid2[0]);
                }
                pictureBox1.Image = img.ToBitmap();
                AddImage(imgROI, "ROI 2");
                treeView1.SelectedNode = null;
            }
            GC.Collect();
        }

        private void TreeView1KeyDown(object sender, KeyEventArgs e)
        {
            if(e.KeyCode == Keys.Delete)
            {
                roi.imgList.Remove(treeView1.SelectedNode.ToString().Replace("TreeNode: ", ""));
                treeView1.SelectedNode.Remove();
                pictureBox1.Image = null;
            }
        }

        private void CloseApplication(object sender, EventArgs e)
        {
            Application.Exit();
        }
    }

    public class ROI
    {
        public Dictionary<string, Image<Bgr, byte>> imgList = new Dictionary<string, Image<Bgr, byte>>();
        public Rectangle rect = Rectangle.Empty;
        public Point startCoord = Point.Empty;
        public Point endCoord = Point.Empty;
        public Point defaultSize = Point.Empty;
        public List<Point> subsequentCentroid = new List<Point>();
        public List<Point> subsequentCentroid2 = new List<Point>();
        public bool isSelect = false;

        public Point FindCentroid(Point[] points)
        {
            float accumulatedArea = 0.0f;
            float centerX = 0.0f;
            float centerY = 0.0f;

            for (int i = 0, j = points.Length - 1; i < points.Length; j = i++)
            {
                float temp = points[i].X * points[j].Y - points[j].X * points[i].Y;
                accumulatedArea += temp;
                centerX += (points[i].X + points[j].X) * temp;
                centerY += (points[i].Y + points[j].Y) * temp;
            }

            if (Math.Abs(accumulatedArea) < 1E-7f)
                return Point.Empty;

            accumulatedArea *= 3f;
            return new Point((int)(centerX / accumulatedArea), (int)(centerY / accumulatedArea));
        }

        public Point FindXYOffset(Point start, Point end)
        {
            return new Point(end.X - start.X, end.Y - start.Y);
        }

        public string CheckTranslationRotation()
        {
            string errorMessage = "";
            Point centroid1 = subsequentCentroid.Last();
            Point centroid2 = subsequentCentroid2.Last();
            Point size = FindXYOffset(subsequentCentroid[0], subsequentCentroid2[0]);
            Point offset = FindXYOffset(subsequentCentroid[0], centroid1);
            if(Math.Abs(offset.X) > 5 || Math.Abs(offset.Y) > 5)
            {
                errorMessage += " Translation: \n X_Diff = " + offset.X + "\n Y_Diff = " + offset.Y + "\r\n"; 
            }
            Point correctEndPosition = new Point(centroid1.X + size.X, centroid1.Y + size.Y);
            double rotationAngle = Math.Round(FindRotationAngle(centroid1, correctEndPosition, centroid2), 2);
            if(rotationAngle > 2) 
            {
                int orientation = FindOrientation(centroid1, correctEndPosition, centroid2);
                string direction = orientation == 1 ? "Anticlockwise" : "Clockwise";
                errorMessage += " Rotation: \n Angle = " + rotationAngle + " deg \n " + direction;
            }
            return errorMessage;
        }

        private double FindRotationAngle(Point startCoord, Point correctEndCoord, Point endCoord)
        {
            double l1 = FindDistance(startCoord, correctEndCoord);
            double l2 = FindDistance(startCoord, endCoord);
            double l3 = FindDistance(endCoord, correctEndCoord);
            double temp = Math.Pow(l1, 2) + Math.Pow(l2, 2) - Math.Pow(l3, 2);
            temp = temp / (2 * l1 * l2);
            return Math.Acos(temp) * 180 / Math.PI;
        }

        private double FindDistance(Point startCoord, Point endCoord)
        {
            double dxSqr = Math.Pow(endCoord.X - startCoord.X, 2);
            double dySqr = Math.Pow(endCoord.Y - startCoord.Y, 2);
            return Math.Sqrt(dxSqr + dySqr);
        }

        private int FindOrientation(Point startCoord, Point correctEndCoord, Point endCoord)
        {
            int val = (correctEndCoord.Y - startCoord.Y) * (endCoord.X - correctEndCoord.X) - (correctEndCoord.X - startCoord.X) * (endCoord.Y - correctEndCoord.Y);
            return (val > 0) ? 1 : 2;
        }
    }
    public class RemoveBorder : ToolStripSystemRenderer
    {
        protected override void OnRenderToolStripBorder(ToolStripRenderEventArgs e)
        {
            //base.OnRenderToolStripBorder(e);
        }
    }
}
