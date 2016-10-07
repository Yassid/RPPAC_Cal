#include <fstream>
#include <iostream>
#include <string>
#include "analyze_PPAC.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <stdlib.h>
#include <stdio.h>

#include "TH1.h"
#include "TF2.h"
#include "TGraph.h"
#include "TCanvas.h"
#include "TH2.h"
#include "TF1.h"
#include "TMath.h"
#include "TApplication.h"
#include "TROOT.h"
#include "Math/Minimizer.h"
#include "Math/Factory.h"
#include "Math/Functor.h"
#include "Math/ParamFunctor.h"

//OpenCV variables
cv::Mat src, src_gray;
cv::Mat dst, detected_edges;

void CannyThreshold(int, void*);
cv::Mat rotate(cv::Mat src, double angle);
Double_t Distance(const double* parameter);
Int_t Minimizer(const char * minName, const char *algoName);
void Reset();

Double_t EvalFit(Double_t *_var, Double_t *_par);

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";
cv::RNG rng(12345);

std::vector<Double_t> q_array1;
std::vector<Double_t> q_array2;
std::vector<Double_t> q_array3;
std::vector<Double_t> q_array4;
std::vector<Double_t> q_norm1;
std::vector<Double_t> q_norm2;
std::vector<Double_t> q_norm3;
std::vector<Double_t> q_norm4;
std::vector<Double_t> q_mean;
std::vector<Double_t> x_d;
std::vector<Double_t> y_d;
std::vector<Double_t> x_u;
std::vector<Double_t> y_u;

TH1I* hadc[4];
TH2D* XY_hist;
TH2D* XY_hist_u;
TH2D* XY_hist_edge;
TH2D* XY_square;

Double_t theta;
Double_t rad_d;
Double_t k1;
Double_t k2;

Double_t norm_fact;
Int_t L; //Square dimension

Double_t* parameter;
Int_t num_par;



Int_t main(int argc, char** argv)
{

  TApplication theApp("App",&argc, argv);

  std::cout<<" Program to calibrate and correct Resistive PPAC data (Y. Ayyad (ayyadlim@nscl.msu.edu) and M. Cortesi (cortesi@nscl.msu.edu)"<<std::endl;

  TCanvas *c1 = new TCanvas();
  c1->Divide(2,2);

  TCanvas *c2 = new TCanvas();
  c2->Divide(1,2);

 TCanvas *c3 = new TCanvas();


  for (Int_t i=0;i<4;i++) hadc[i] = new TH1I(Form("hadc[%i]",i),Form("hadc%i",i),1000,0,4000);

  XY_hist = new TH2D("XY_hist","XY_hist",1000,-50,50,1000,-50,50);
  XY_hist_u = new TH2D("XY_hist_u","XY_hist_u",1000,-50,50,1000,-50,50);
  XY_hist_edge = new TH2D("XY_hist_edge","XY_hist_edge",1000,-50,50,1000,-50,50);
  XY_square = new TH2D("XY_square","XY_square",1000,-50,50,1000,-50,50);

  TString filename = "../run106.txt";
  std::ifstream fData(filename,std::ios::binary);
  UInt_t header = 0;
  UInt_t adc[4];
  q_mean.reserve(4);
  for(Int_t i=0;i<4;i++)q_mean[i]=0.0;
  Int_t q_cnt = 0;
  std::string strbuff;
  Int_t n= 4;

  // Flat field calibration
  Bool_t kFlatCal =1;


  Char_t * buffer = new char [n];

  std::string line;
  Int_t nEve = 0;

  norm_fact = 100.0;
  L = 110.0; //Square dimension

  while(std::getline(fData, line)){
    nEve++;
    //if(nEve%100000==0)std::cout<<nEve<<std::endl;
    std::vector<char> writable(line.begin(), line.end());
    writable.push_back('\0');
    Int_t num = writable.at(3)-'0';



     if( std::isdigit(writable.at(3))){
          strbuff =  line.substr(0,4);
          header = std::stoi(strbuff);
          if(header&0x0006){
              //std::cout<<header<<"      "<<strbuff<<std::endl;
              adc[0] = strtol(line.substr(10,4).c_str(),NULL,16); //NB:: substr takes the sapce into account
              adc[1] = strtol(line.substr(15,4).c_str(),NULL,16);
              adc[2] = strtol(line.substr(20,4).c_str(),NULL,16);
              adc[3] = strtol(line.substr(25,4).c_str(),NULL,16);
              hadc[0]->Fill(adc[0]);
              hadc[1]->Fill(adc[1]);
              hadc[2]->Fill(adc[2]);
              hadc[3]->Fill(adc[3]);
              q_cnt++;
              q_mean[0]+=adc[0];
              q_mean[1]+=adc[1];
              q_mean[2]+=adc[2];
              q_mean[3]+=adc[3];
              q_array1.push_back(adc[0]);
              q_array2.push_back(adc[1]);
              q_array3.push_back(adc[2]);
              q_array4.push_back(adc[3]);



              //std::cout<<adc[0]<<"     "<<line.substr(10,4).c_str()<<std::endl;
              //std::cout<<adc[1]<<"     "<<line.substr(15,4).c_str()<<std::endl;
              //std::cout<<adc[2]<<"     "<<line.substr(20,4).c_str()<<std::endl;
              //std::cout<<adc[3]<<"     "<<line.substr(25,4).c_str()<<std::endl;
              //std::cout<<line<<std::endl;
          }

     }

  }


  for(Int_t i=0;i<4;i++)
    q_mean[i]/=q_cnt;


   if(kFlatCal ==1){

	std::cout<<" Using previous calibrated data "<<std::endl; 
	q_mean[0] = 244.4;
	q_mean[1] = 234.6;
	q_mean[2] = 159.6;
	q_mean[3] = 162.2;

   }


  // Analysis and Reconstruction
  theta=0.0;
  rad_d=0.0;
  k1 =0.0010;
  k2 =0.0000;

  num_par=2;
  parameter = new Double_t[num_par];
  parameter[0] = k1;
  parameter[1] = k2;


  Double_t min_dist = Distance(parameter);
  std::cout<<" Minimum distance : "<<min_dist<<std::endl;
  //Minimizer("Minuit","Migrad");  // Temporary disabled

	Double_t parx[13] = {0.2591E+01,0.2141E+01,0.2982E-01,0.3252E-01,-.2188E-01,0.1485E-01,-.4089E-04,-.1306E-01,0.1150E-03,0.7579E-02,0.2000E-04,-.3663E-04,-.1430E-04}; 
	Double_t var[2] = {1.0,2.0};
	EvalFit(var,parx);
		


  c1->cd(1);
  hadc[0]->SetLineColor(kRed);
  hadc[0]->Draw();
  c1->cd(2);
  hadc[1]->SetLineColor(kBlack);
  hadc[1]->Draw();
  c1->cd(3);
  hadc[2]->SetLineColor(kBlue);
  hadc[2]->Draw();
  c1->cd(4);
  hadc[3]->SetLineColor(kGreen);
  hadc[3]->Draw();

  c2->cd(1);
  XY_hist->Draw("zcol");
  c2->cd(2);
  XY_hist_u->Draw("zcol");

  c3->cd();
  XY_hist_u->Draw("zcol");
  XY_hist_edge->SetMarkerSize(0.5);
  XY_hist_edge->Draw("SAME");
  XY_square->SetMarkerColor(kRed);
  XY_square->SetMarkerSize(0.5);
  XY_square->Draw("SAMES");
  //c3->SaveAs("ppac.eps");


  fData.close();
  theApp.Run();
  return 0;


}

Int_t Minimizer(const char * minName, const char *algoName){


	std::cout<<" Parameter 0 : "<<parameter[0]<<std::endl;
	ROOT::Math::Minimizer* min = ROOT::Math::Factory::CreateMinimizer(minName, algoName);
	min->SetMaxFunctionCalls(1000000);
         min->SetMaxIterations(5);
	min->SetTolerance(0.001);
	min->SetPrecision(0.001);
	min->SetPrintLevel(1);

	ROOT::Math::Functor f(&Distance,num_par);
	min->SetFunction(f);
        min->SetLimitedVariable(0,"p0",parameter[0],0.00001,0,0.0030);
	//min->SetLimitedVariable(1,"p1",parameter[1],0.00001,0,0.0030);
        min->SetFixedVariable(1,"p1",parameter[1]);
	min->Minimize();



}

void Reset()
{

  q_norm1.clear();
  q_norm2.clear();
  q_norm3.clear();
  q_norm4.clear();
  x_d.clear();
  y_d.clear();
  XY_hist->Reset();
  x_u.clear();
  y_u.clear();
  XY_hist_u->Reset();
  XY_hist_edge->Reset();
  XY_square->Reset();

}


Double_t Distance(const double* parameter)
{

   Reset();
   std::ofstream kineStr;
   kineStr.open("data_Qcal.txt");


    for(Int_t i=0;i<q_array1.size();i++){

          q_norm1.push_back(q_array1.at(i)*norm_fact/q_mean[0]);
          q_norm2.push_back(q_array2.at(i)*norm_fact/q_mean[1]);
          q_norm3.push_back(q_array3.at(i)*norm_fact/q_mean[2]);
          q_norm4.push_back(q_array4.at(i)*norm_fact/q_mean[3]);

          Double_t q_tot = q_norm1.at(i) + q_norm2.at(i) + q_norm3.at(i) + q_norm4.at(i);

            x_d.push_back(  (L/2.0)*( (q_norm2.at(i) + q_norm3.at(i)) - (q_norm1.at(i) + q_norm4.at(i))             )/ q_tot            );
            y_d.push_back(  (L/2.0)*( (q_norm3.at(i) + q_norm4.at(i)) - (q_norm1.at(i) + q_norm2.at(i))             )/ q_tot            );

            XY_hist->Fill(x_d.at(i),y_d.at(i));

	    //kineStr<<x_d.at(i)<<"	"<<y_d.at(i)<<std::endl;


    }


	Double_t X_mean = XY_hist->GetMean(1);
        Double_t Y_mean = XY_hist->GetMean(2);


     for(Int_t i=0;i<q_array1.size();i++){

            
            x_d.at(i)-=X_mean;
            y_d.at(i)-=Y_mean;

	    kineStr<<x_d.at(i)<<"	"<<y_d.at(i)<<std::endl;

            theta = TMath::ATan2(y_d.at(i),x_d.at(i));
            rad_d = TMath::Sqrt( TMath::Power(x_d.at(i),2) + TMath::Power(y_d.at(i),2)      );

            x_u.push_back( (rad_d/(1+parameter[0]*TMath::Power(rad_d,2)  +   parameter[1]*TMath::Power(rad_d,4)  ) )*TMath::Cos(theta)      );
            y_u.push_back( (rad_d/(1+parameter[0]*TMath::Power(rad_d,2)  +   parameter[1]*TMath::Power(rad_d,4)  ) )*TMath::Sin(theta)      );

            XY_hist_u->Fill(x_u.at(i),y_u.at(i));


  }

  		cv::Mat src(1000,1000,cv::DataType<double>::type);
 	         for(Int_t i=0;i<1000;i++)
		   for(Int_t j=0;j<1000;j++){
				 if(XY_hist_u->GetBinContent(i,j)>0) src.at<double>(i,j) = 100.0;
				 else src.at<double>(i,j) = 0.0;
		   }

		 cv::imwrite("ppac.png",src);

		 //This works
		 src = cv::imread("ppac.png");
		 //cv::namedWindow( "window", cv::WINDOW_AUTOSIZE );
		 //cv::imshow("window", src );
	         //cv::waitKey(0);


		 dst.create( src.size(), src.type() );
		 ///src.convertTo(src,CV_8U);
		 ///cv::cvtColor( src, src_gray, CV_BGR2GRAY );
		 //cv::namedWindow( window_name, CV_WINDOW_AUTOSIZE );


		 cv::Mat gray,edge, draw;
		 cv::blur( src,src, cv::Size(3,3) );
		 cv::Canny( src, edge, 50, 150, 3);

		/// edge.convertTo(draw, CV_8U);
    		 /*cv::namedWindow("image", CV_WINDOW_AUTOSIZE);
                 cv::imshow("image", edge);
 		 cv::waitKey(0);*/

		 cv::imwrite("ppac_edge.png",edge);



		 std::cout<<" Size of the Edge matrix : "<<edge.size()<<" with "<<edge.channels()<<" RGB channels. Continuous?: "
		 <<edge.isContinuous()<<" Image Step : "<<edge.step<<" Image Rows : "<<edge.rows<<std::endl;

		 edge = rotate(edge,90);

		 unsigned char *input = (unsigned char*)(edge.data);
		 Double_t dist_acc = 0.0;
		 Int_t dist_cnt = 0;


		   for(Int_t i=0;i<1000;i++){
			   const double* edge_ptr = edge.ptr<double>(i);
		   	for(Int_t j=0;j<1000;j++)
				{
				 if(input[edge.step*j+i]>0){
					 XY_hist_edge->SetBinContent(i,1000-j,input[edge.step*j+i]);
					 double x = XY_hist_edge->GetXaxis()->GetBinCenter(i);
  					 double y = XY_hist_edge->GetYaxis()->GetBinCenter(1000-j);
					 dist_acc+=TMath::Sqrt(TMath::Power(x,2) + TMath::Power(y,2));
					 dist_cnt++;
				}
				//if(edge.at<double>(i,j)>0) std::cout<<" i : "<<i<<" j : "<<j<<" Value : "<<edge.at<double>(i,j)<<std::endl;
				}

		 }

		 //Constructing a square
		 Int_t sq_size = 10.0; //half distance in cm
		 Int_t sq_pnt = 200;
		 Double_t dist_acc_sqr = 0.0;

		 for(Int_t i =0;i<sq_pnt;i++ ){

				Double_t coord = -sq_size + i*2*sq_size/(Double_t)sq_pnt;
				Double_t coord_bin=XY_square->GetXaxis()->FindBin(coord);
				Double_t bin_low=XY_square->GetYaxis()->FindBin(-sq_size);
				Double_t bin_up=XY_square->GetYaxis()->FindBin(sq_size);
				XY_square->SetBinContent(coord_bin,bin_low,1);
				XY_square->SetBinContent(coord_bin,bin_up,1);
				XY_square->SetBinContent(bin_low,coord_bin,1);
				XY_square->SetBinContent(bin_up,coord_bin,1);
			  dist_acc_sqr+=TMath::Sqrt(TMath::Power(coord,2) + TMath::Power(sq_size,2));


		 }



		 std::vector<std::vector<cv::Point> > contours;
 		 std::vector<cv::Vec4i> hierarchy;
		 //cv::threshold(edge,edge,127,255,0);
		 cv::findContours( edge, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

		 std::cout<<" Found "<<contours.size()<<" contours "<<std::endl;
		 std::cout<<" Mean distance from the contour to the center : "<<dist_acc/(Double_t) dist_cnt<<std::endl;
		 std::cout<<" Mean distance from the square to the center : "<<dist_acc_sqr/(Double_t)sq_pnt<<std::endl;

		 Double_t min_dist = TMath::Abs(dist_acc/(Double_t) dist_cnt - dist_acc_sqr/(Double_t)sq_pnt);

		 return min_dist;

}

void CannyThreshold(int, void*)
{
  /// Reduce noise with a kernel 3x3
  cv::blur( src_gray, detected_edges, cv::Size(3,3) );

  /// Canny detector
  cv::Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  dst = cv::Scalar::all(0);

  src.copyTo( dst, detected_edges);
  if(!dst.empty()) cv::imshow( window_name, dst );
 }

cv::Mat rotate(cv::Mat src, double angle)
{
    cv::Mat dst;
    cv::Point2f pt(src.cols/2., src.rows/2.);
    cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
    cv::warpAffine(src, dst, r, cv::Size(src.cols, src.rows));
    return dst;
}

Double_t EvalFit(Double_t *_var, Double_t *_p)
{

	
	TF1 f1("f1","x");
	TF1 f2("f2","x");
	TF2 f3("f3","x*y");
        TF1 f4("f4","x*x");
	TF1 f5("f5","x*x");
	TF1 f6("f6","x*x*x");	
	TF2 f7("f7","x*y*y");
	TF2 f8("f7","x*x*y");
	TF1 f9("f9","x*x*x");
	TF2 f10("f10","x*y*y*y*y");
	TF2 f11("f11","x*y*y*y");
	TF2 f12("f12","x*x*x*y*y");


	TF2 fsum("fsum",[&](double *var, double *p){ return p[0] + p[1]*f1(var[0]) + p[2]*f2(var[1]) + p[3]*f3(var[0],var[1]) 
	+ p[4]*f4(var[1]) + p[5]*f5(var[0]) + p[6]*f6(var[1]) + p[7]*f7(var[0],var[1]) + p[8]*f8(var[0],var[1]) + p[9]*f9(var[0]) 
	+ p[10]*f10(var[0],var[1]) + p[11]*f11(var[0],var[1]) + p[12]*f12(var[0],var[1])  ; },-50,50,-50,50,3);

	Double_t result = fsum.EvalPar(_var,_p);
	//std::cout<<result<<std::endl;	

	return result;

}


///////////
 // Draw the contour with CV. Not useful for the moment
		 /*std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
  		 std::vector<cv::Rect> boundRect( contours.size() );
  		 std::vector<cv::Point2f>center( contours.size() );
  		 std::vector<float>radius( contours.size() );


		  for( int i = 0; i < contours.size(); i++ )
     		  {
			cv::approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 3, true );
       			boundRect[i] = boundingRect( cv::Mat(contours_poly[i]) );
       			cv::minEnclosingCircle( (cv::Mat)contours_poly[i], center[i], radius[i] );
     		  }

		  cv::Mat drawing = cv::Mat::zeros( edge.size(), CV_8UC3 );
		  for( int i = 0; i< contours.size(); i++ )
		     {
		       cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		       cv::drawContours( drawing, contours_poly, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point() );
		       cv::rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
		       //cv::circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
		     }
		 */
		 ////////////////////////////////////////////////////////////////////////

		 //cv::namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
  		 //cv::imshow( "Contours", drawing );
		 //cv::waitKey(0);

		 /*cv::createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );
		 CannyThreshold(0, 0);
		 dst = cv::Scalar::all(0);
                 src.copyTo( dst, detected_edges);
		 cv::imshow( window_name, dst );*/


